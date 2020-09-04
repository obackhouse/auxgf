# Options
CC=gcc
CDR=$(shell pwd)

# Flags
CFLAGS=-Wall -Wextra -std=c11 -O3 -ffast-math
OFLAGS=-fPIC -fopenmp -lopenblas -lm
SOFLAGS=-shared -lgomp -lm -lopenblas -ggdb3

all: agf2

clean: clean_agf2

clean_agf2:
	rm -f $(CDR)/libagf2/*.o
	rm -f $(CDR)/libagf2/*.so

optragf2:
	$(CC) $(CFLAGS) $(OFLAGS) -c $(CDR)/libagf2/optragf2.c -o $(CDR)/libagf2/optragf2.o

optuagf2:
	$(CC) $(CFLAGS) $(OFLAGS) -c $(CDR)/libagf2/optuagf2.c -o $(CDR)/libagf2/optuagf2.o

agf2: optragf2 optuagf2
	$(CC) $(CFLAGS) $(SOFLAGS) $(CDR)/libagf2/optragf2.o $(CDR)/libagf2/optuagf2.o -o $(CDR)/libagf2/agf2.so
