# Makefile for Parallel Data Analytics Project

CC = gcc
NVCC = nvcc
MPICC = mpicc
CFLAGS = -Wall -O3 -std=c11
OMPFLAGS = -fopenmp
CUDAFLAGS = -O3 -arch=sm_75
LDFLAGS = -lm -lpthread

# Source files
SEQ_SRC = sequential.c
PTHREADS_SRC = pthreads.c
OPENMP_SRC = openmp.c
MPI_SRC = mpi.c
CUDA_SRC = cuda.cu

# Executables
SEQ_EXE = sequential
PTHREADS_EXE = pthreads
OPENMP_EXE = openmp
MPI_EXE = mpi
CUDA_EXE = cuda

# Default target
all: $(SEQ_EXE) $(PTHREADS_EXE) $(OPENMP_EXE) $(MPI_EXE) $(CUDA_EXE)

# Sequential implementation
$(SEQ_EXE): $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $(SEQ_EXE) $(SEQ_SRC) $(LDFLAGS)

# Pthreads implementation
$(PTHREADS_EXE): $(PTHREADS_SRC)
	$(CC) $(CFLAGS) -o $(PTHREADS_EXE) $(PTHREADS_SRC) $(LDFLAGS)

# OpenMP implementation
$(OPENMP_EXE): $(OPENMP_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $(OPENMP_EXE) $(OPENMP_SRC) $(LDFLAGS)

# MPI implementation
$(MPI_EXE): $(MPI_SRC)
	$(MPICC) $(CFLAGS) -o $(MPI_EXE) $(MPI_SRC) $(LDFLAGS)

# CUDA implementation
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(CUDAFLAGS) -o $(CUDA_EXE) $(CUDA_SRC)

# Clean build artifacts
clean:
	rm -f $(SEQ_EXE) $(PTHREADS_EXE) $(OPENMP_EXE) $(MPI_EXE) $(CUDA_EXE)
	rm -f *.o

# Run all experiments
run: all
	@echo "Running performance experiments..."
	@bash run_experiments.sh

.PHONY: all clean run

