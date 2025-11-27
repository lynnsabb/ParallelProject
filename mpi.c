// MPI Implementation: Statistical Feature Extraction with distributed-memory parallelism

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }

    char buffer[BUFFER_SIZE];
    if (!fgets(buffer, BUFFER_SIZE, file)) {
        fclose(file);
        return 0;
    }

    int feature_indices[] = {2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    char *feature_names[] = {
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
        int col = 0, valid = 1, feat_idx = 0;
        double values[MAX_FEATURES];

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
            for (int i = 0; i < MAX_FEATURES; i++) {
                ds->data[i][ds->num_records] = values[i];
            }
            ds->num_records++;
        }
    }

    fclose(file);
    return 1;
}

double compute_mean(double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum / n;
}

double compute_stddev(double *data, int n, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / n);
}

double compute_correlation(double *x, double *y, int n) {
    double mean_x = compute_mean(x, n);
    double mean_y = compute_mean(y, n);
    double std_x = compute_stddev(x, n, mean_x);
    double std_y = compute_stddev(y, n, mean_y);

    if (std_x == 0.0 || std_y == 0.0) {
        return 0.0;
    }

    double sum_product = 0.0;
    for (int i = 0; i < n; i++) {
        sum_product += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return sum_product / (n * std_x * std_y);
}

void parallel_correlation_matrix(Dataset *ds, double **corr_matrix, 
                                 int rank, int size) {
    int num_features = ds->num_features;
    int rows_per_process = num_features / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? num_features : (rank + 1) * rows_per_process;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < num_features; j++) {
            corr_matrix[i][j] = compute_correlation(
                ds->data[i], ds->data[j], ds->num_records);
        }
    }

    if (rank == 0) {
        MPI_Status status;
        for (int p = 1; p < size; p++) {
            int p_start = p * rows_per_process;
            int p_end = (p == size - 1) ? num_features : (p + 1) * rows_per_process;
            int p_rows = p_end - p_start;
            for (int i = 0; i < p_rows; i++) {
                MPI_Recv(corr_matrix[p_start + i], num_features, MPI_DOUBLE, 
                        p, 0, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        for (int i = start_row; i < end_row; i++) {
            MPI_Send(corr_matrix[i], num_features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void parallel_statistical_moments(Dataset *ds, double *means, double *variances, 
                                  double *skewness, int rank, int size) {
    int num_features = ds->num_features;
    int features_per_process = num_features / size;
    int start_feat = rank * features_per_process;
    int end_feat = (rank == size - 1) ? num_features : (rank + 1) * features_per_process;

    for (int f = start_feat; f < end_feat; f++) {
        means[f] = compute_mean(ds->data[f], ds->num_records);
        double std = compute_stddev(ds->data[f], ds->num_records, means[f]);
        variances[f] = std * std;
        double sum_cubed = 0.0;
        for (int i = 0; i < ds->num_records; i++) {
            double normalized = (ds->data[f][i] - means[f]) / (std + 1e-10);
            sum_cubed += normalized * normalized * normalized;
        }
        skewness[f] = sum_cubed / ds->num_records;
    }

    if (rank == 0) {
        MPI_Status status;
        for (int p = 1; p < size; p++) {
            int p_start = p * features_per_process;
            int p_end = (p == size - 1) ? num_features : (p + 1) * features_per_process;
            int p_count = p_end - p_start;
            MPI_Recv(means + p_start, p_count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(variances + p_start, p_count, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(skewness + p_start, p_count, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, &status);
        }
    } else {
        int count = end_feat - start_feat;
        MPI_Send(means + start_feat, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(variances + start_feat, count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(skewness + start_feat, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    
    MPI_Bcast(means, num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(variances, num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(skewness, num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void perform_analysis(Dataset *ds, int rank, int size) {
    double **corr_matrix = (double **)malloc(ds->num_features * sizeof(double *));
    for (int i = 0; i < ds->num_features; i++) {
        corr_matrix[i] = (double *)malloc(ds->num_features * sizeof(double));
    }

    double *means = (double *)malloc(ds->num_features * sizeof(double));
    double *variances = (double *)malloc(ds->num_features * sizeof(double));
    double *skewness = (double *)malloc(ds->num_features * sizeof(double));

    if (rank == 0) printf("Computing correlation matrix with %d processes...\n", size);
    parallel_correlation_matrix(ds, corr_matrix, rank, size);

    if (rank == 0) printf("Computing statistical moments with %d processes...\n", size);
    parallel_statistical_moments(ds, means, variances, skewness, rank, size);

    if (rank == 0) {
        printf("\n=== Sample Results ===\n");
        printf("Correlation between %s and %s: %.4f\n", 
               ds->feature_names[0], ds->feature_names[1], corr_matrix[0][1]);
        printf("Mean of %s: %.4f\n", ds->feature_names[0], means[0]);
        printf("Variance of %s: %.4f\n", ds->feature_names[0], variances[0]);
        printf("Skewness of %s: %.4f\n", ds->feature_names[0], skewness[0]);
    }

    for (int i = 0; i < ds->num_features; i++) free(corr_matrix[i]);
    free(corr_matrix);
    free(means);
    free(variances);
    free(skewness);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) {
        printf("Usage: mpirun -np <num_processes> %s <dataset.csv>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Dataset ds;
    if (!load_dataset(argv[1], &ds)) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Loaded %d records with %d features\n", ds.num_records, ds.num_features);
        printf("Running with %d MPI processes\n", size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    perform_analysis(&ds, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        double cpu_time = end_time - start_time;
        printf("\n=== Performance ===\n");
        printf("MPI execution time (%d processes): %.4f seconds\n", size, cpu_time);
        printf("Records processed: %d\n", ds.num_records);
        printf("Features analyzed: %d\n", ds.num_features);
    }

    for (int i = 0; i < ds.num_features; i++) free(ds.data[i]);
    MPI_Finalize();
    return 0;
}

