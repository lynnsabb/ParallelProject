# Parallel Data Analytics Across Architectures

## Project Overview

This project implements statistical feature extraction and correlation matrix computation on the Australian Weather Dataset using five different parallel programming paradigms:

1. **Sequential** (C) - Baseline implementation
2. **Pthreads** - CPU-level thread parallelism
3. **OpenMP** - Shared-memory parallelism
4. **MPI** - Distributed-memory parallelism
5. **CUDA** - GPU parallelism

## Algorithm

The project performs:
- **Correlation Matrix Computation**: Computes Pearson correlation coefficients between all pairs of numerical weather features
- **Statistical Moments**: Calculates mean, variance, and skewness for each feature

**Why this algorithm is suitable for parallelization:**
- Correlation computation is O(n × m²) and can be parallelized across feature pairs
- Statistical moment calculations are independent across features
- High computational complexity provides sufficient workload for parallelization
- Dataset with 58,236 records (after preprocessing) provides measurable performance differences

## Dataset

**Australian Weather Dataset** (`weatherAUS.csv`)
- Source: Kaggle (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- Original records: ~145,000 weather observations
- Records after preprocessing: 58,236 (records with missing values are excluded)
- Features: 16 numerical features (temperature, humidity, pressure, wind speed, etc.)

## Compilation

### Prerequisites
- GCC compiler with C11 support
- OpenMP support (usually included with GCC)
- MPI implementation (MS-MPI for Windows, OpenMPI or MPICH for Linux/Mac)
- CUDA Toolkit (version 10.0+ for GPU implementation)
- Make utility (optional, for Linux/Mac)

### Build Instructions

```bash
# Build all implementations
make

# Or build individually:
gcc -Wall -O3 -std=c11 -o sequential sequential.c -lm
gcc -Wall -O3 -std=c11 -o pthreads pthreads.c -lm -lpthread
gcc -Wall -O3 -std=c11 -fopenmp -o openmp openmp.c -lm
mpicc -Wall -O3 -std=c11 -o mpi mpi.c -lm
nvcc -O3 -arch=sm_75 -o cuda cuda.cu
```

### Windows Compilation

```powershell
# Sequential
gcc -Wall -O3 -std=c11 -o sequential.exe sequential.c -lm

# Pthreads (requires pthreads library for Windows)
gcc -Wall -O3 -std=c11 -o pthreads.exe pthreads.c -lm -lpthread

# OpenMP
gcc -Wall -O3 -std=c11 -fopenmp -o openmp.exe openmp.c -lm

# MPI (Windows - use x64 Native Tools Command Prompt)
cl /O2 /EHsc /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" mpi.c /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib /out:mpi.exe

# CUDA
nvcc -O3 -arch=sm_75 -o cuda.exe cuda.cu
```

## Execution

### Sequential
```bash
./sequential weatherAUS.csv
```

### Pthreads
```bash
./pthreads weatherAUS.csv <num_threads>
# Example: ./pthreads weatherAUS.csv 4
```

### OpenMP
```bash
./openmp weatherAUS.csv <num_threads> <schedule>
# Example: ./openmp weatherAUS.csv 8 static
# Schedules: static, dynamic, guided
```

### MPI
```bash
mpirun -np <num_processes> ./mpi weatherAUS.csv
# Example: mpirun -np 4 ./mpi weatherAUS.csv
```

### CUDA
```bash
./cuda weatherAUS.csv <kernel_type> [block_size]
# Example: ./cuda weatherAUS.csv tiled 256
# Kernel types: simple, tiled
# Block sizes: 128, 256, 512 (default: 256)
```

## Performance Evaluation

Run the automated performance evaluation script:

```bash
# Linux/Mac
bash run_experiments.sh

# Windows
run_experiments.bat
```

Results will be saved to `results/performance_data.txt` and `results/sample_results.txt`

## Expected Configurations for Testing

- **Pthreads**: 2, 4, 8 threads
- **OpenMP**: 2, 4, 8 threads with static, dynamic, guided schedules
- **MPI**: 2, 4, 8 processes
- **CUDA**: simple and tiled kernel configurations

## Project Structure

```
.
├── sequential.c          # Sequential implementation
├── pthreads.c            # Pthreads implementation
├── openmp.c              # OpenMP implementation
├── mpi.c                 # MPI implementation
├── cuda.cu               # CUDA implementation
├── weatherAUS.csv        # Dataset
├── Makefile              # Build configuration
├── run_experiments.sh    # Performance evaluation script (Linux/Mac)
├── run_experiments.bat   # Performance evaluation script (Windows)
├── README.md             # This file
├── REPORT.md             # Project report
└── results/              # Performance results directory
    ├── performance_data.txt  # Detailed performance data
    └── sample_results.txt    # Sample execution results
```

## CUDA Optimizations

1. **Shared Memory Utilization**: Reduces global memory access latency
2. **Memory Coalescing**: Optimizes global memory access patterns
3. **Tiling Techniques**: Improves cache locality
4. **Block/Grid Size Tuning**: Optimizes thread block configuration
5. **Warp Divergence Avoidance**: Minimizes branch divergence

## Results Analysis

The performance evaluation generates:
- Runtime comparisons across all implementations
- Speedup calculations relative to sequential baseline
- Scalability analysis for varying thread/process counts
- Comparison tables for different configurations

## Team Member Roles

- **Noor Chouman**: Implemented the Sequential and OpenMP versions, organized the report, and analyzed performance results.
- **Jana Ellweis**: Implemented the Pthreads and MPI versions, ran scalability experiments, and contributed to result interpretation.
- **Lynn Sabbagh**: Implemented and optimized CUDA kernels, handled dataset preprocessing, and contributed to the discussion section.

All members participated in testing, debugging, and preparing the final presentation.

## Notes

- The dataset must be placed in the project root directory
- Ensure sufficient memory for loading the full dataset
- CUDA implementation requires a compatible NVIDIA GPU


