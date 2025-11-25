@echo off
REM Performance Data Collection Script
REM Runs all implementations and saves results

echo ========================================
echo Performance Data Collection
echo ========================================
echo.

set OUTPUT=results\performance_data.txt
echo Performance Results - %date% %time% > %OUTPUT%
echo ======================================== >> %OUTPUT%
echo. >> %OUTPUT%

echo [1/5] Sequential...
echo === Sequential === >> %OUTPUT%
.\sequential.exe weatherAUS.csv >> %OUTPUT% 2>&1
echo. >> %OUTPUT%

echo [2/5] Pthreads...
echo === Pthreads === >> %OUTPUT%
echo Threads: 2 >> %OUTPUT%
.\pthreads.exe weatherAUS.csv 2 >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 4 >> %OUTPUT%
.\pthreads.exe weatherAUS.csv 4 >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 8 >> %OUTPUT%
.\pthreads.exe weatherAUS.csv 8 >> %OUTPUT% 2>&1
echo. >> %OUTPUT%

echo [3/5] OpenMP...
echo === OpenMP === >> %OUTPUT%
echo Threads: 2, Schedule: static >> %OUTPUT%
.\openmp.exe weatherAUS.csv 2 static >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 4, Schedule: static >> %OUTPUT%
.\openmp.exe weatherAUS.csv 4 static >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 8, Schedule: static >> %OUTPUT%
.\openmp.exe weatherAUS.csv 8 static >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 8, Schedule: dynamic >> %OUTPUT%
.\openmp.exe weatherAUS.csv 8 dynamic >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Threads: 8, Schedule: guided >> %OUTPUT%
.\openmp.exe weatherAUS.csv 8 guided >> %OUTPUT% 2>&1
echo. >> %OUTPUT%

echo [4/5] MPI...
echo === MPI === >> %OUTPUT%
echo Processes: 2 >> %OUTPUT%
mpiexec -n 2 .\mpi.exe weatherAUS.csv >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Processes: 4 >> %OUTPUT%
mpiexec -n 4 .\mpi.exe weatherAUS.csv >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Processes: 8 >> %OUTPUT%
mpiexec -n 8 .\mpi.exe weatherAUS.csv >> %OUTPUT% 2>&1
echo. >> %OUTPUT%

echo [5/5] CUDA...
echo === CUDA === >> %OUTPUT%
echo Kernel: simple >> %OUTPUT%
.\cuda.exe weatherAUS.csv simple >> %OUTPUT% 2>&1
echo. >> %OUTPUT%
echo Kernel: tiled >> %OUTPUT%
.\cuda.exe weatherAUS.csv tiled >> %OUTPUT% 2>&1
echo. >> %OUTPUT%

echo.
echo ========================================
echo Performance data collection complete!
echo Results saved to: %OUTPUT%
echo ========================================
pause

