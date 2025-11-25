@echo off
REM Performance Evaluation Script for Windows
REM Runs all implementations with varying configurations

set DATASET=weatherAUS.csv
set RESULTS_DIR=results
set OUTPUT_FILE=%RESULTS_DIR%\performance_results.txt

echo === Parallel Data Analytics Performance Evaluation === > %OUTPUT_FILE%
echo Dataset: %DATASET% >> %OUTPUT_FILE%
echo Date: %date% %time% >> %OUTPUT_FILE%
echo. >> %OUTPUT_FILE%

REM Sequential baseline
echo Running Sequential Implementation...
echo === Sequential === >> %OUTPUT_FILE%
sequential.exe %DATASET% >> %OUTPUT_FILE% 2>&1
echo. >> %OUTPUT_FILE%

REM Pthreads with varying thread counts
echo Running Pthreads Implementation...
echo === Pthreads === >> %OUTPUT_FILE%
for %%t in (2 4 8) do (
    echo Testing with %%t threads...
    echo Threads: %%t >> %OUTPUT_FILE%
    pthreads.exe %DATASET% %%t >> %OUTPUT_FILE% 2>&1
)
echo. >> %OUTPUT_FILE%

REM OpenMP with varying thread counts
echo Running OpenMP Implementation...
echo === OpenMP === >> %OUTPUT_FILE%
for %%t in (2 4 8) do (
    for %%s in (static dynamic guided) do (
        echo Testing with %%t threads, %%s schedule...
        echo Threads: %%t, Schedule: %%s >> %OUTPUT_FILE%
        openmp.exe %DATASET% %%t %%s >> %OUTPUT_FILE% 2>&1
    )
)
echo. >> %OUTPUT_FILE%

REM CUDA with different kernel configurations
echo Running CUDA Implementation...
echo === CUDA === >> %OUTPUT_FILE%
for %%k in (simple tiled) do (
    echo Testing with %%k kernel...
    echo Kernel: %%k >> %OUTPUT_FILE%
    cuda.exe %DATASET% %%k >> %OUTPUT_FILE% 2>&1
)
echo. >> %OUTPUT_FILE%

echo Performance evaluation complete. Results saved to %OUTPUT_FILE%

