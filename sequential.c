/**
 * Sequential Implementation: Statistical Feature Extraction
 * 
 * This program computes:
 * 1. Correlation matrix between numerical weather features
 * 2. Statistical moments (mean, variance, skewness)
 * 3. Matrix normalization operations
 * 
 * Algorithm: Pearson correlation coefficient computation
 * Time Complexity: O(n * m^2) where n = number of records, m = number of features
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_FEATURES 16
#define MAX_RECORDS 150000
#define BUFFER_SIZE 1024

// Numerical feature indices in CSV
typedef struct {
    double *data[MAX_FEATURES];
    int num_features;
    int num_records;
    char feature_names[MAX_FEATURES][32];
} Dataset;

/**
 * Parse CSV and extract numerical features
 */
int load_dataset(const char *filename, Dataset *ds) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }

    char buffer[BUFFER_SIZE];
    
    // Read header
    if (!fgets(buffer, BUFFER_SIZE, file)) {
        fclose(file);
        return 0;
    }

    // Initialize feature arrays
    ds->num_features = 0;
    int feature_indices[] = {2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}; // Column indices
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

    // Read data rows
    while (fgets(buffer, BUFFER_SIZE, file) && ds->num_records < MAX_RECORDS) {
        char *token = strtok(buffer, ",");
        int col = 0;
        int valid = 1;
        double values[MAX_FEATURES];
        int feat_idx = 0;

        while (token != NULL && col < 23) {
            // Check if this column is a numerical feature we want
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
    printf("Loaded %d records with %d features\n", ds->num_records, ds->num_features);
    return 1;
}

/**
 * Compute mean of a feature vector
 */
double compute_mean(double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

/**
 * Compute standard deviation of a feature vector
 */
double compute_stddev(double *data, int n, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / n);
}

/**
 * Compute Pearson correlation coefficient between two features
 */
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

/**
 * Compute correlation matrix for all feature pairs
 */
void compute_correlation_matrix(Dataset *ds, double **corr_matrix) {
    printf("Computing correlation matrix...\n");
    for (int i = 0; i < ds->num_features; i++) {
        for (int j = 0; j < ds->num_features; j++) {
            corr_matrix[i][j] = compute_correlation(ds->data[i], ds->data[j], ds->num_records);
        }
    }
}

/**
 * Compute statistical moments (mean, variance, skewness) for all features
 */
void compute_statistical_moments(Dataset *ds, double *means, double *variances, double *skewness) {
    printf("Computing statistical moments...\n");
    
    for (int f = 0; f < ds->num_features; f++) {
        means[f] = compute_mean(ds->data[f], ds->num_records);
        double std = compute_stddev(ds->data[f], ds->num_records, means[f]);
        variances[f] = std * std;

        // Compute skewness
        double sum_cubed = 0.0;
        for (int i = 0; i < ds->num_records; i++) {
            double normalized = (ds->data[f][i] - means[f]) / (std + 1e-10);
            sum_cubed += normalized * normalized * normalized;
        }
        skewness[f] = sum_cubed / ds->num_records;
    }
}

/**
 * Matrix-vector multiplication for feature normalization
 */
void normalize_features(Dataset *ds, double *means, double *stddevs) {
    printf("Normalizing features...\n");
    for (int f = 0; f < ds->num_features; f++) {
        double mean = means[f];
        double std = stddevs[f];
        if (std < 1e-10) std = 1.0;

        for (int i = 0; i < ds->num_records; i++) {
            ds->data[f][i] = (ds->data[f][i] - mean) / std;
        }
    }
}

/**
 * Main computation function
 */
void perform_analysis(Dataset *ds) {
    // Allocate correlation matrix
    double **corr_matrix = (double **)malloc(ds->num_features * sizeof(double *));
    for (int i = 0; i < ds->num_features; i++) {
        corr_matrix[i] = (double *)malloc(ds->num_features * sizeof(double));
    }

    // Allocate statistical arrays
    double *means = (double *)malloc(ds->num_features * sizeof(double));
    double *variances = (double *)malloc(ds->num_features * sizeof(double));
    double *skewness = (double *)malloc(ds->num_features * sizeof(double));
    double *stddevs = (double *)malloc(ds->num_features * sizeof(double));

    // Compute correlation matrix
    compute_correlation_matrix(ds, corr_matrix);

    // Compute statistical moments
    compute_statistical_moments(ds, means, variances, skewness);
    for (int i = 0; i < ds->num_features; i++) {
        stddevs[i] = sqrt(variances[i]);
    }

    // Normalize features (matrix operations)
    normalize_features(ds, means, stddevs);

    // Print sample results
    printf("\n=== Sample Results ===\n");
    printf("Correlation between %s and %s: %.4f\n", 
           ds->feature_names[0], ds->feature_names[1], corr_matrix[0][1]);
    printf("Mean of %s: %.4f\n", ds->feature_names[0], means[0]);
    printf("Variance of %s: %.4f\n", ds->feature_names[0], variances[0]);
    printf("Skewness of %s: %.4f\n", ds->feature_names[0], skewness[0]);

    // Cleanup
    for (int i = 0; i < ds->num_features; i++) {
        free(corr_matrix[i]);
    }
    free(corr_matrix);
    free(means);
    free(variances);
    free(skewness);
    free(stddevs);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <dataset.csv>\n", argv[0]);
        return 1;
    }

    Dataset ds;
    if (!load_dataset(argv[1], &ds)) {
        return 1;
    }

    clock_t start = clock();
    perform_analysis(&ds);
    clock_t end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\n=== Performance ===\n");
    printf("Sequential execution time: %.4f seconds\n", cpu_time);
    printf("Records processed: %d\n", ds.num_records);
    printf("Features analyzed: %d\n", ds.num_features);

    // Cleanup
    for (int i = 0; i < ds.num_features; i++) {
        free(ds.data[i]);
    }

    return 0;
}

