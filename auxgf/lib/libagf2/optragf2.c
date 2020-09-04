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


void build_part_loop_rhf(double *ixq,
                         double *qja,
                         double *ei,
                         double *ea,
                         uint32_t nphys,
                         uint32_t nocc,
                         uint32_t nvir,
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
     *  ixq : double array
     *      density fitting integral block (ix|Q)
     *  qja : double array
     *      density fitting integral block (Q|ja)
     *  ei : double array
     *      occupied QMO energies
     *  ea : double array
     *      virtual QMO energies
     *  nphys : uint32_t
     *      number of physical degrees of freedom
     *  nocc : uint32_t
     *      number of occupied QMOs
     *  nvir : uint32_t
     *      number of virtual QMOs
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
     *      output array for V . e . V^\dagger (nphys * nphys)
     */

    // Setup timer if required
#if PROFILE
    INIT_TIMER;
    GET_TIME(t0);
#endif

    // Begin the parallel scope
#pragma omp parallel
    {

    // Initialise DGEMM variables
    char transa, transb;
    uint32_t m, n, k, lda, ldb, ldc;
    double alpha, beta;
    double *a, *b, *c;

    // Allocate the memory for the temporary qa slice
    double *qa = calloc(naux*nvir, sizeof(double));

    // Allocate the memory for the energies
    double *eja = calloc(nocc*nvir, sizeof(double));

    // Allocate the memory for the integral blocks
    double *xia = calloc(nphys*nocc*nvir, sizeof(double));
    double *xja = calloc(nphys*nocc*nvir, sizeof(double));
    double *exja = calloc(nphys*nocc*nvir, sizeof(double));

    // Allocate thread-private vv and vev
    double *vv_thread = calloc(nphys*nphys, sizeof(double));
    double *vev_thread = calloc(nphys*nphys, sizeof(double));

#if PROFILE
    RECORD_TIME(0);
#endif

    // Start the loop
#pragma omp for
    for (size_t i = istart; i < iend; i++) {
        // Prepare the qa slice
        for (size_t q = 0; q < naux; q++) {
            for (size_t a = 0; a < nvir; a++) {
                qa[IDX2(q,a,nvir)] = qja[IDX3(q,i,a,nocc,nvir)];
            }
        }

#if PROFILE
        RECORD_TIME(1);
#endif

        // Prepare the xia array (actually ixa for now - use the xja 
        // working array to store this before transposing)
        a = &(qa[0]);
        b = &(ixq[0]);
        c = &(xja[0]);
        transa = 'N';
        transb = 'N';
        m = nvir;
        n = nocc*nphys;
        k = naux;
        lda = nvir;
        ldb = naux;
        ldc = nvir;
        alpha = 1.0;
        beta = 0.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(2);
#endif

        // Reshape xia (swap the first two dimensions)
        for (size_t x = 0, idx1 = 0; x < nphys; x++) {
            for (size_t j = 0; j < nocc; j++) {
                for (size_t a = 0; a < nvir; a++, idx1++) {
                    xia[idx1] = xja[IDX3(j,x,a,nphys,nvir)];
                }
            }
        }

#if PROFILE
        RECORD_TIME(3);
#endif

        // Reset xja to zero
        for (size_t idx = 0; idx < (nphys*nocc*nvir); idx++) {
            xja[idx] = 0.0;
        }

#if PROFILE
        RECORD_TIME(4);
#endif
        
        // Prepare the xja array
        a = &(qja[0]);
        b = &(ixq[IDX3(i,0,0,nphys,naux)]);
        c = &(xja[0]);
        transa = 'N';
        transb = 'N';
        m = nocc*nvir;
        n = nphys;
        k = naux;
        lda = nocc*nvir;
        ldb = naux;
        ldc = nocc*nvir;
        alpha = 1.0;
        beta = 0.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(5);
#endif

        // Build the energies
        for (size_t j = 0; j < nocc; j++) {
            for (size_t a = 0; a < nvir; a++) {
                eja[IDX2(j,a,nvir)] = ei[i] + ei[j] - ea[a];
            }
        }

#if PROFILE
        RECORD_TIME(6);
#endif

        // Build the Coulomb contribution to the vv matrix
        a = &(xja[0]);
        b = &(xja[0]);
        c = &(vv_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc*nvir;
        lda = nocc*nvir;
        ldb = nocc*nvir;
        ldc = nphys;
        alpha = 2.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(7);
#endif

        // Build the exchange contribution to the vv matrix
        a = &(xia[0]);
        b = &(xja[0]);
        c = &(vv_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc*nvir;
        lda = nocc*nvir;
        ldb = nocc*nvir;
        ldc = nphys;
        alpha = -1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(8);
#endif

        // Prepare the energy-weighted xja array
        for (size_t x = 0; x < nphys; x++) {
            for (size_t ja = 0; ja < (nocc*nvir); ja++) {
                exja[IDX2(x,ja,nocc*nvir)] = xja[IDX2(x,ja,nocc*nvir)] * eja[ja];
            }
        }

#if PROFILE
        RECORD_TIME(9);
#endif

        // Build the Coulomb contribution to the vev matrix
        a = &(xja[0]);
        b = &(exja[0]);
        c = &(vev_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc*nvir;
        lda = nocc*nvir;
        ldb = nocc*nvir;
        ldc = nphys;
        alpha = 2.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(10);
#endif

        // Build the exchange contribution to the vev matrix
        a = &(xia[0]);
        b = &(exja[0]);
        c = &(vev_thread[0]);
        transa = 'T';
        transb = 'N';
        m = nphys;
        n = nphys;
        k = nocc*nvir;
        lda = nocc*nvir;
        ldb = nocc*nvir;
        ldc = nphys;
        alpha = -1.0;
        beta = 1.0;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if PROFILE
        RECORD_TIME(11);
#endif
    }

    // Free up the work arrays
    free(qa);
    free(eja);
    free(xia);
    free(xja);
    free(exja);

    // Reduce the thread-private vv and vev onto the output arrays
#pragma omp critical
    for (size_t xy = 0; xy < (nphys*nphys); xy++) {
        vv[xy] += vv_thread[xy];
        vev[xy] += vev_thread[xy];
    }

    // Free the thread-private vv and vev
    free(vv_thread);
    free(vev_thread);

#if PROFILE
    RECORD_TIME(12);
#endif

    // End the parallel scope
    }

#if PROFILE
    printf("%12s %12s\n",   "section", "time (s)");
    printf("%12s %12s\n",   "------------", "------------");
    printf("%12s %12.6f\n", "calloc", timings[0]);
    printf("%12s %12.6f\n", "qa slice", timings[1]);
    printf("%12s %12.6f\n", "ixq . qa", timings[2]);
    printf("%12s %12.6f\n", "ixa -> xia", timings[3]);
    printf("%12s %12.6f\n", "xja = {0}", timings[4]);
    printf("%12s %12.6f\n", "xq . qja", timings[5]);
    printf("%12s %12.6f\n", "eja", timings[6]);
    printf("%12s %12.6f\n", "xja . jax", timings[7]);
    printf("%12s %12.6f\n", "xja . iax", timings[8]);
    printf("%12s %12.6f\n", "exja", timings[9]);
    printf("%12s %12.6f\n", "exja . jax", timings[10]);
    printf("%12s %12.6f\n", "exja . iax", timings[11]);
    printf("%12s %12.6f\n", "free", timings[12]);
#endif
}
