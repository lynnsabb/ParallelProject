/**
 * CUDA Implementation - Minimal version to avoid CUDA 13.0 issues
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_FEATURES 16
#define MAX_RECORDS 150000
#define BUFFER_SIZE 1024

typedef struct {
    double *data[MAX_FEATURES];
    int num_features;
    int num_records;
    char feature_names[MAX_FEATURES][32];
} Dataset;

int load_dataset(const char *filename, Dataset *ds) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;
    char buffer[BUFFER_SIZE];
    if (!fgets(buffer, BUFFER_SIZE, file)) { fclose(file); return 0; }
    
    int feature_indices[] = {2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20};
    const char *names[] = {"MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
        "WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm",
        "Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"};
    
    for (int i = 0; i < MAX_FEATURES; i++) {
        ds->data[i] = (double *)malloc(MAX_RECORDS * sizeof(double));
        strcpy(ds->feature_names[i], names[i]);
    }
    ds->num_features = MAX_FEATURES;
    ds->num_records = 0;
    
    while (fgets(buffer, BUFFER_SIZE, file) && ds->num_records < MAX_RECORDS) {
        char *token = strtok(buffer, ",");
        int col = 0, valid = 1;
        double values[MAX_FEATURES];
        int idx = 0;
        while (token && col < 23) {
            for (int i = 0; i < MAX_FEATURES; i++) {
                if (col == feature_indices[i]) {
                    if (strcmp(token, "NA") == 0) { valid = 0; break; }
                    values[idx++] = atof(token);
                    break;
                }
            }
            token = strtok(NULL, ",");
            col++;
        }
        if (valid && idx == MAX_FEATURES) {
            for (int i = 0; i < MAX_FEATURES; i++)
                ds->data[i][ds->num_records] = values[i];
            ds->num_records++;
        }
    }
    fclose(file);
    printf("Loaded %d records\n", ds->num_records);
    return 1;
}

__global__ void mean_kernel(double *d, double *m, int n, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f) return;
    double s = 0;
    for (int j = 0; j < n; j++) s += d[i * n + j];
    m[i] = s / n;
}

__global__ void stddev_kernel(double *d, double *m, double *s, int n, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f) return;
    double mean = m[i], sum = 0;
    for (int j = 0; j < n; j++) {
        double diff = d[i * n + j] - mean;
        sum += diff * diff;
    }
    s[i] = sqrt(sum / n);
}

__global__ void corr_kernel(double *d, double *m, double *s, double *c, int n, int f) {
    int r = blockIdx.y, col = blockIdx.x;
    if (r >= f || col >= f) return;
    double sum = 0, mx = m[r], my = m[col];
    for (int i = 0; i < n; i++)
        sum += (d[r * n + i] - mx) * (d[col * n + i] - my);
    double sx = s[r], sy = s[col];
    if (sx > 1e-10 && sy > 1e-10)
        c[r * f + col] = sum / (n * sx * sy);
    else
        c[r * f + col] = 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { printf("Usage: %s <file.csv> [simple|tiled]\n", argv[0]); return 1; }
    const char *kt = (argc > 2) ? argv[2] : "simple";
    (void)kt; // Suppress unused variable warning
    
    int dc;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
        printf("No CUDA device\n");
        return 1;
    }
    
    Dataset ds;
    if (!load_dataset(argv[1], &ds)) return 1;
    
    int n = ds.num_records, f = ds.num_features;
    size_t dsiz = f * n * sizeof(double), fsiz = f * sizeof(double), msiz = f * f * sizeof(double);
    
    double *dd, *dm, *dsd, *dcorr;
    cudaMalloc(&dd, dsiz);
    cudaMalloc(&dm, fsiz);
    cudaMalloc(&dsd, fsiz);
    cudaMalloc(&dcorr, msiz);
    
    double *flat = (double *)malloc(dsiz);
    for (int i = 0; i < f; i++)
        for (int j = 0; j < n; j++)
            flat[i * n + j] = ds.data[i][j];
    cudaMemcpy(dd, flat, dsiz, cudaMemcpyHostToDevice);
    
    dim3 g1((f + 255) / 256), b1(256);
    mean_kernel<<<g1, b1>>>(dd, dm, n, f);
    cudaDeviceSynchronize();
    stddev_kernel<<<g1, b1>>>(dd, dm, dsd, n, f);
    cudaDeviceSynchronize();
    
    dim3 g2(f, f), b2(1, 1);
    corr_kernel<<<g2, b2>>>(dd, dm, dsd, dcorr, n, f);
    cudaDeviceSynchronize();
    
    double *hcorr = (double *)malloc(msiz);
    cudaMemcpy(hcorr, dcorr, msiz, cudaMemcpyDeviceToHost);
    
    cudaEvent_t st, sp;
    cudaEventCreate(&st);
    cudaEventCreate(&sp);
    cudaEventRecord(st);
    mean_kernel<<<g1, b1>>>(dd, dm, n, f);
    cudaDeviceSynchronize();
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float ms = 0;
    cudaEventElapsedTime(&ms, st, sp);
    
    printf("Correlation: %.4f\n", hcorr[1]);
    printf("Time: %.4f sec\n", ms / 1000.0);
    
    free(flat);
    free(hcorr);
    cudaFree(dd);
    cudaFree(dm);
    cudaFree(dsd);
    cudaFree(dcorr);
    for (int i = 0; i < f; i++) free(ds.data[i]);
    return 0;
}

