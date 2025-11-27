#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_FEATURES 16
#define MAX_RECORDS 150000
#define BUFFER_SIZE 1024
#define BLOCK_SIZE 256
#define TILE_SIZE 16

typedef struct {
    double *data[MAX_FEATURES];
    int num_features;
    int num_records;
    char feature_names[MAX_FEATURES][32];
} Dataset;

int load_dataset(const char *filename, Dataset *ds) {
    FILE *file = fopen(filename, "r");
    if (!file) { printf("Error: Cannot open file %s\n", filename); return 0; }
    char buffer[BUFFER_SIZE];
    if (!fgets(buffer, BUFFER_SIZE, file)) { fclose(file); return 0; }

    int feature_indices[] = {2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    const char *feature_names[] = {
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
    };

    for (int i = 0; i < MAX_FEATURES; i++) {
        ds->data[i] = (double *)malloc(MAX_RECORDS * sizeof(double));
        strcpy(ds->feature_names[i], feature_names[i]);
    }
    ds->num_features = MAX_FEATURES;
    ds->num_records = 0;

    while (fgets(buffer, BUFFER_SIZE, file) && ds->num_records < MAX_RECORDS) {
        char *token = strtok(buffer, ",");
        int col = 0;
        int valid = 1;
        double values[MAX_FEATURES];
        int feat_idx = 0;

        while (token != NULL && col < 23) {
            for (int i = 0; i < MAX_FEATURES; i++) {
                if (col == feature_indices[i]) {
                    if (strcmp(token, "NA") == 0 || strlen(token) == 0) {
                        valid = 0;
                        break;
                    }
                    values[feat_idx++] = atof(token);
                    break;
                }
            }
            token = strtok(NULL, ",");
            col++;
        }

        if (valid && feat_idx == MAX_FEATURES) {
            for (int i = 0; i < MAX_FEATURES; i++) ds->data[i][ds->num_records] = values[i];
            ds->num_records++;
        }
    }

    fclose(file);
    printf("Loaded %d records with %d features\n", ds->num_records, ds->num_features);
    return 1;
}

__global__ void compute_mean_kernel(double *data, double *means, int n, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[idx * n + i];
    }
    means[idx] = sum / n;
}

__global__ void compute_mean_kernel_shared(double *data, double *means, int n, int num_features) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    if (idx < num_features) {
        for (int i = 0; i < n; i++) sum += data[idx * n + i];
    }
    sdata[tid] = (idx < num_features) ? sum / n : 0.0;
    __syncthreads();
    if (idx < num_features) means[idx] = sdata[tid];
}

__global__ void compute_stddev_kernel(double *data, double *means, double *stddevs, int n, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    double mean = means[idx], sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[idx * n + i] - mean;
        sum_sq += diff * diff;
    }
    stddevs[idx] = sqrt(sum_sq / n);
}

__global__ void compute_correlation_kernel_simple(double *data, double *means, double *stddevs, 
                                                   double *corr_matrix, int n, int num_features) {
    int row = blockIdx.y, col = blockIdx.x;
    if (row >= num_features || col >= num_features) return;
    double sum = 0.0, mean_x = means[row], mean_y = means[col];
    for (int i = 0; i < n; i++) sum += (data[row * n + i] - mean_x) * (data[col * n + i] - mean_y);
    double std_x = stddevs[row], std_y = stddevs[col];
    if (std_x > 1e-10 && std_y > 1e-10) corr_matrix[row * num_features + col] = sum / (n * std_x * std_y);
    else corr_matrix[row * num_features + col] = 0.0;
}

__global__ void compute_correlation_kernel_tiled(double *data, double *means, double *stddevs,
                                                  double *corr_matrix, int n, int num_features) {
    __shared__ double tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_y[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    double sum = 0.0;
    double mean_x = (row < num_features) ? means[row] : 0.0;
    double mean_y = (col < num_features) ? means[col] : 0.0;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int idx_x = tile * TILE_SIZE + tx;
        int idx_y = tile * TILE_SIZE + ty;
        
        if (row < num_features && idx_x < n) {
            tile_x[ty][tx] = data[row * n + idx_x] - mean_x;
        } else {
            tile_x[ty][tx] = 0.0;
        }
        
        if (col < num_features && idx_y < n) {
            tile_y[ty][tx] = data[col * n + idx_y] - mean_y;
        } else {
            tile_y[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        if (row < num_features && col < num_features) {
            for (int k = 0; k < TILE_SIZE && (tile * TILE_SIZE + k) < n; k++) {
                sum += tile_x[ty][k] * tile_y[k][tx];
            }
        }
        __syncthreads();
    }
    
    if (row < num_features && col < num_features) {
        double std_x = stddevs[row], std_y = stddevs[col];
        if (std_x > 1e-10 && std_y > 1e-10) {
            corr_matrix[row * num_features + col] = sum / (n * std_x * std_y);
        } else {
            corr_matrix[row * num_features + col] = 0.0;
        }
    }
}

void perform_analysis(Dataset *ds, const char *kernel_type, int block_size) {
    int n = ds->num_records;
    int num_features = ds->num_features;
    
    double *d_data, *d_means, *d_stddevs, *d_corr_matrix;
    size_t data_size = num_features * n * sizeof(double);
    size_t feature_size = num_features * sizeof(double);
    size_t matrix_size = num_features * num_features * sizeof(double);
    
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_means, feature_size);
    cudaMalloc(&d_stddevs, feature_size);
    cudaMalloc(&d_corr_matrix, matrix_size);
    
    double *h_data_flat = (double *)malloc(data_size);
    for (int f = 0; f < num_features; f++)
        for (int i = 0; i < n; i++) h_data_flat[f * n + i] = ds->data[f][i];
    cudaMemcpy(d_data, h_data_flat, data_size, cudaMemcpyHostToDevice);
    
    dim3 grid_mean((num_features + block_size - 1) / block_size);
    dim3 block_mean(block_size);
    compute_mean_kernel_shared<<<grid_mean, block_mean>>>(d_data, d_means, n, num_features);
    cudaDeviceSynchronize();
    
    compute_stddev_kernel<<<grid_mean, block_mean>>>(d_data, d_means, d_stddevs, n, num_features);
    cudaDeviceSynchronize();
    
    if (strcmp(kernel_type, "tiled") == 0) {
        dim3 grid_corr((num_features + TILE_SIZE - 1) / TILE_SIZE, (num_features + TILE_SIZE - 1) / TILE_SIZE);
        dim3 block_corr(TILE_SIZE, TILE_SIZE);
        compute_correlation_kernel_tiled<<<grid_corr, block_corr>>>(d_data, d_means, d_stddevs, d_corr_matrix, n, num_features);
    } else {
        dim3 grid_corr(num_features, num_features);
        dim3 block_corr(block_size, 1);
        compute_correlation_kernel_simple<<<grid_corr, block_corr>>>(d_data, d_means, d_stddevs, d_corr_matrix, n, num_features);
    }
    cudaDeviceSynchronize();
    
    double *h_means = (double *)malloc(feature_size);
    double *h_corr_matrix = (double *)malloc(matrix_size);
    cudaMemcpy(h_means, d_means, feature_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_corr_matrix, d_corr_matrix, matrix_size, cudaMemcpyDeviceToHost);
    
    printf("\n=== Sample Results ===\n");
    printf("Correlation between %s and %s: %.4f\n", ds->feature_names[0], ds->feature_names[1], h_corr_matrix[0 * num_features + 1]);
    printf("Mean of %s: %.4f\n", ds->feature_names[0], h_means[0]);
    
    free(h_data_flat);
    free(h_means);
    free(h_corr_matrix);
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_stddevs);
    cudaFree(d_corr_matrix);
}

int main(int argc, char *argv[]) {
    if (argc < 3) { 
        printf("Usage: %s <dataset.csv> <kernel_type> [block_size]\n", argv[0]); 
        printf("Kernel types: tiled, simple\n");
        printf("Block sizes: 128, 256, 512 (default: 256)\n");
        return 1; 
    }
    const char *kernel_type = argv[2];
    int block_size = (argc > 3) ? atoi(argv[3]) : 256;
    if (block_size != 128 && block_size != 256 && block_size != 512) {
        printf("Warning: Block size %d not standard, using 256\n", block_size);
        block_size = 256;
    }
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) { printf("Error: No CUDA devices found\n"); return 1; }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);
    printf("Kernel type: %s\n", kernel_type);
    printf("Block size: %d threads\n", block_size);
    
    Dataset ds;
    if (!load_dataset(argv[1], &ds)) return 1;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    perform_analysis(&ds, kernel_type, block_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double cpu_time = milliseconds / 1000.0;
    
    printf("\n=== Performance ===\n");
    printf("CUDA execution time (%s kernel, block_size=%d): %.4f seconds\n", kernel_type, block_size, cpu_time);
    printf("Records processed: %d\n", ds.num_records);
    printf("Features analyzed: %d\n", ds.num_features);
    
    for (int i = 0; i < ds.num_features; i++) free(ds.data[i]);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
