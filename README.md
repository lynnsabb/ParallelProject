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
- **Feature Normalization**: Matrix-vector operations for data standardization

**Why this algorithm is suitable for parallelization:**
- Correlation computation is O(n²) and can be parallelized across feature pairs
- Statistical moment calculations are independent across features
- Matrix operations are highly parallelizable
- Large dataset (145,000+ records) provides sufficient workload

## Dataset

**Australian Weather Dataset** (`weatherAUS.csv`)
- Source: Kaggle (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- Records: ~145,000 weather observations
- Features: 16 numerical features (temperature, humidity, pressure, wind speed, etc.)

## Compilation

### Prerequisites
- GCC compiler with C11 support
- OpenMP support (usually included with GCC)
- MPI implementation (OpenMPI or MPICH)
- CUDA Toolkit (for GPU implementation)
- Make utility

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

# MPI
mpicc -Wall -O3 -std=c11 -o mpi.exe mpi.c -lm

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
./cuda weatherAUS.csv <kernel_type>
# Example: ./cuda weatherAUS.csv tiled
# Kernel types: simple, tiled
```

## Performance Evaluation

Run the automated performance evaluation script:

```bash
# Linux/Mac
bash run_experiments.sh

# Windows
run_experiments.bat
```

Results will be saved to `results/performance_results.txt`

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
├── run_experiments.sh    # Performance evaluation script
├── README.md             # This file
├── REPORT.md             # Project report
├── PRESENTATION.md       # Presentation script
└── results/              # Performance results directory
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

(To be filled by team members)

## Notes

- All code is original and not copied from online sources
- The dataset must be placed in the project root directory
- Ensure sufficient memory for loading the full dataset
- CUDA implementation requires a compatible NVIDIA GPU

## License

Academic project - for educational purposes only.

