#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<omp.h>
#include<stdio.h>
#include<time.h>

#define IDX2(a,b,nb) ((a)*(nb) + (b))
#define IDX3(a,b,c,nb,nc) ((a)*(nb)*(nc) + (b)*(nc) + (c))

#define PROFILE 0
#define INIT_TIMER clock_t t0, t1; double timings[13]; t0 = clock()
#define GET_TIME(t) (t) = clock()
#define RECORD_TIME(idx) GET_TIME(t1); timings[(idx)] = (double)(t1 - t0) / CLOCKS_PER_SEC; GET_TIME(t0)


extern void dgemm_(char *transa,
                   char *transb,
                   uint32_t *m,
                   uint32_t *n,
                   uint32_t *k,
                   double *alpha,
                   double *a,
                   uint32_t *lda,
                   double *b,
                   uint32_t *ldb,
                   double *beta,
                   double *c,
                   uint32_t *ldc);

void build_part_loop_uhf(double *ixq_a,
                         double *qja_a,
                         double *qja_b,
                         double *ei_a,
                         double *ei_b,
                         double *ea_a,
                         double *ea_b,
                         uint32_t nphys,
                         uint32_t nocc_a,
                         uint32_t nocc_b,
                         uint32_t nvir_a,
                         uint32_t nvir_b,
                         uint32_t naux,
                         uint32_t istart,
                         uint32_t iend,
                         double *vv,
                         double *vev)
{
    /*
     *  Compute the VV^\dagger and VeV^\dagger matrices for a single
     *  MPI rank with OpenMP parallelism.
     *
     *  All DGEMM calls are flipped for C memory layout.
     *
     *  Parameters
     *  ----------
     *  ixq_a : double array
     *      density fitting integral block (ix|Q) for alpha spin
     *  qja_a : double array
     *      density fitting integral block (Q|ja) for alpha spin
     *  qja_b : double array
     *      density fitting integral block (Q|ja) for beta spin
     *  ei_a : double array
     *      occupied QMO energies for alpha spin
     *  ei_b : double array
     *      occupied QMO energies for beta spin
     *  ea_a : double array
     *      virtual QMO energies for alpha spin
     *  ea_b : double array
     *      virtual QMO energies for beta spin
     *  nphys : uint32_t
     *      number of physical degrees of freedom
     *  nocc_a : uint32_t
     *      number of occupied QMOs for alpha spin
     *  nocc_b : uint32_t
     *      number of occupied QMOs for beta spin
     *  nvir_a : uint32_t
     *      number of virtual QMOs for alpha spin
     *  nvir_b : uint32_t
     *      number of virtual QMOs for beta spin
     *  naux : uint32_t
     *      number of auxiliary degrees of freedom in DF integrals
     *  istart : uint32_t
     *      index i to start at on current process
     *  iend : uint32_t
     *      index i to end at on current process (non-inclusive)
     *
     *  Returns
     *  -------
     *  vv : double array
     *      output array for V . V^\dagger (nphys * nphys)
     *  vev : double array
     *      output array for V . e . V^\dagger
     */

    // Begin the parallel scope
#pragma omp parallel
    {

    // Initialise DGEMM variables
    char transa, transb;
    uint32_t m, n, k, lda, ldb, ldc;
    double alpha, beta;
    double *a, *b, *c;

    // Allocate the memory for the temporary qa slice
    double *qa_a = calloc(naux*nvir_a, sizeof(double));

    // Allocate the memory for the energies
    double *eja_aa = calloc(nocc_a*nvir_a, sizeof(double));
    double *eja_ab = calloc(nocc_b*nvir_b, sizeof(double));

    // Allocate the memory for the integral blocks
    double *xia_aa = calloc(nphys*nocc_a*nvir_a, sizeof(double));
    double *xja_aa = calloc(nphys*nocc_a*nvir_a, sizeof(double));
    double *xja_ab = calloc(nphys*nocc_b*nvir_b, sizeof(double));
    double *exja_aa = calloc(nphys*nocc_a*nvir_a, sizeof(double));
    double *exja_ab = calloc(nphys*nocc_b*nvir_b, sizeof(double));

    // Allocate thread-private vv and vev
    double *vv_thread = calloc(nphys*nphys, sizeof(double));
    double *vev_thread = calloc(nphys*nphys, sizeof(double));

    // Start the loop
#pragma omp for
    for (size_t i = istart; i < iend; i++) {
        // Prepare the qa slice
        for (size_t q = 0; q < naux; q++) {
            for (size_t a = 0; a < nvir_a; a++) {
                qa_a[IDX2(q,a,nvir_a)] = qja_a[IDX3(q,i,a,nocc_a,nvir_a)];
            }
        }

        // Prepare the xia_aa array (actually ixa for now - use the xja_aa
        // working array to store this before transposing)
        a = &(qa_a[0]);
        b = &(ixq_a[0]);
        c = &(xja_aa[0]);
        transa = 'N';
        transb = 'N';
        m = nvir_a;
        n = nocc_a*nphys;
        k = naux;
        lda = nvir_a;
        ldb = naux;
        ldc = nvir_a;
        alpha = 1.0;
        beta = 0.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Reshape xia_aa (swap the first two dimensions)
        for (size_t x = 0, idx1 = 0; x < nphys; x++) {
            for (size_t j = 0; j < nocc_a; j++) {
                for (size_t a = 0; a < nvir_a; a++, idx1++) {
                    xia_aa[idx1] = xja_aa[IDX3(j,x,a,nphys,nvir_a)];
                }
            }
        }

        // Reset xja_aa to zero
        for (size_t idx = 0; idx < (nphys*nocc_a*nvir_a); idx++) {
            xja_aa[idx] = 0.0;
        }

        // Prepare the xja_aa array
        a = &(qja_a[0]);
        b = &(ixq_a[IDX3(i,0,0,nphys,naux)]);
        c = &(xja_aa[0]);
        transa = 'N';
        transb = 'N';
        m = nocc_a*nvir_a;
        n = nphys;
        k = naux;
        lda = nocc_a*nvir_a;
        ldb = naux;
        ldc = nocc_a*nvir_a;
        alpha = 1.0;
        beta = 0.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Prepare the xja_ab array
        a = &(qja_b[0]);
        b = &(ixq_a[IDX3(i,0,0,nphys,naux)]);
        c = &(xja_ab[0]);
        transa = 'N';
        transb = 'N';
        m = nocc_b*nvir_b;
        n = nphys;
        k = naux;
        lda = nocc_b*nvir_b;
        ldb = naux;
        ldc = nocc_b*nvir_b;
        alpha = 1.0;
        beta = 0.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Build the energies
        for (size_t j = 0; j < nocc_a; j++) {
            for (size_t a = 0; a < nvir_a; a++) {
                eja_aa[IDX2(j,a,nvir_a)] = ei_a[i] + ei_a[j] - ea_a[a];
            }
        }

        for (size_t j = 0; j < nocc_b; j++) {
            for (size_t a = 0; a < nvir_b; a++) {
                eja_ab[IDX2(j,a,nvir_b)] = ei_a[i] + ei_b[j] - ea_b[a];
            }
        }

        // Build the alpha-alpha Coulomb contribution to the vv matrix
        a = &(xja_aa[0]);
        b = &(xja_aa[0]);
        c = &(vv_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_a*nvir_a;
        lda = nocc_a*nvir_a;
        ldb = nocc_a*nvir_a;
        ldc = nphys;
        alpha = 1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Build the alpha-beta Coulomb contribution to the vv matrix
        a = &(xja_ab[0]);
        b = &(xja_ab[0]);
        c = &(vv_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_b*nvir_b;
        lda = nocc_b*nvir_b;
        ldb = nocc_b*nvir_b;
        ldc = nphys;
        alpha = 1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Build the exchange contribution to the vv matrix:
        a = &(xia_aa[0]);
        b = &(xja_aa[0]);
        c = &(vv_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_a*nvir_a;
        lda = nocc_a*nvir_a;
        ldb = nocc_a*nvir_a;
        ldc = nphys;
        alpha = -1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Prepare the energy-weighted xja_aa and xja_ab arrays
        for (size_t x = 0; x < nphys; x++) {
            for (size_t ja = 0; ja < (nocc_a*nvir_a); ja++) {
                exja_aa[IDX2(x,ja,nocc_a*nvir_a)] = xja_aa[IDX2(x,ja,nocc_a*nvir_a)] * eja_aa[ja];
            }
            for (size_t ja = 0; ja < (nocc_b*nvir_b); ja++) {
                exja_ab[IDX2(x,ja,nocc_b*nvir_b)] = xja_ab[IDX2(x,ja,nocc_b*nvir_b)] * eja_ab[ja];
            }
        }

        // Build the alpha-alpha Coulomb contribution to the vev matrix
        a = &(xja_aa[0]);
        b = &(exja_aa[0]);
        c = &(vev_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_a*nvir_a;
        lda = nocc_a*nvir_a;
        ldb = nocc_a*nvir_a;
        ldc = nphys;
        alpha = 1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Build the alpha-beta Coulomb contribution to the vev matrix
        a = &(xja_ab[0]);
        b = &(exja_ab[0]);
        c = &(vev_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_b*nvir_b;
        lda = nocc_b*nvir_b;
        ldb = nocc_b*nvir_b;
        ldc = nphys;
        alpha = 1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

        // Build the exchange contribution to the vev matrix:
        a = &(xia_aa[0]);
        b = &(exja_aa[0]);
        c = &(vev_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc_a*nvir_a;
        lda = nocc_a*nvir_a;
        ldb = nocc_a*nvir_a;
        ldc = nphys;
        alpha = -1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    // Free up the work arrays
    free(qa_a);
    free(eja_aa);
    free(eja_ab);
    free(xia_aa);
    free(xja_aa);
    free(xja_ab);
    free(exja_aa);
    free(exja_ab);

    // Reduce the thread-private vv and vev onto the output arrays
#pragma omp critical
    for (size_t xy = 0; xy < (nphys*nphys); xy++) {
        vv[xy] += vv_thread[xy];
        vev[xy] += vev_thread[xy];
    }

    // Free the thread-private vv and vev
    free(vv_thread);
    free(vev_thread);

    // End the parallel scope
    }
}
