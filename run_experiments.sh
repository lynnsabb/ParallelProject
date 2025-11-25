#!/bin/bash

# Performance Evaluation Script
# Runs all implementations with varying configurations and collects results

DATASET="weatherAUS.csv"
RESULTS_DIR="results"
OUTPUT_FILE="$RESULTS_DIR/performance_results.txt"

echo "=== Parallel Data Analytics Performance Evaluation ===" > $OUTPUT_FILE
echo "Dataset: $DATASET" >> $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Sequential baseline
echo "Running Sequential Implementation..."
echo "=== Sequential ===" >> $OUTPUT_FILE
./sequential $DATASET 2>&1 | tee -a $OUTPUT_FILE
SEQUENTIAL_TIME=$(./sequential $DATASET 2>&1 | grep "Sequential execution time" | awk '{print $4}')
echo "" >> $OUTPUT_FILE

# Pthreads with varying thread counts
echo "Running Pthreads Implementation..."
echo "=== Pthreads ===" >> $OUTPUT_FILE
for threads in 2 4 8; do
    echo "Testing with $threads threads..."
    echo "Threads: $threads" >> $OUTPUT_FILE
    ./pthreads $DATASET $threads 2>&1 | grep "Pthreads execution time" | tee -a $OUTPUT_FILE
done
echo "" >> $OUTPUT_FILE

# OpenMP with varying thread counts and schedules
echo "Running OpenMP Implementation..."
echo "=== OpenMP ===" >> $OUTPUT_FILE
for threads in 2 4 8; do
    for schedule in static dynamic guided; do
        echo "Testing with $threads threads, $schedule schedule..."
        echo "Threads: $threads, Schedule: $schedule" >> $OUTPUT_FILE
        ./openmp $DATASET $threads $schedule 2>&1 | grep "OpenMP execution time" | tee -a $OUTPUT_FILE
    done
done
echo "" >> $OUTPUT_FILE

# MPI with varying process counts
echo "Running MPI Implementation..."
echo "=== MPI ===" >> $OUTPUT_FILE
for processes in 2 4 8; do
    echo "Testing with $processes processes..."
    echo "Processes: $processes" >> $OUTPUT_FILE
    mpirun -np $processes ./mpi $DATASET 2>&1 | grep "MPI execution time" | tee -a $OUTPUT_FILE
done
echo "" >> $OUTPUT_FILE

# CUDA with different kernel configurations
echo "Running CUDA Implementation..."
echo "=== CUDA ===" >> $OUTPUT_FILE
for kernel in simple tiled; do
    echo "Testing with $kernel kernel..."
    echo "Kernel: $kernel" >> $OUTPUT_FILE
    ./cuda $DATASET $kernel 2>&1 | grep "CUDA execution time" | tee -a $OUTPUT_FILE
done
echo "" >> $OUTPUT_FILE

echo "Performance evaluation complete. Results saved to $OUTPUT_FILE"

