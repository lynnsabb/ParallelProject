@echo off
REM Compile CUDA with Visual Studio environment setup

echo Setting up Visual Studio x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not set up Visual Studio environment
    pause
    exit /b 1
)

echo.
echo Adding CUDA to PATH...
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin

echo.
echo Compiling CUDA implementation (cuda.cu)...
echo.

echo [1/3] Trying with O0 (no optimizations)...
nvcc -O0 -o cuda.exe cuda.cu
if %ERRORLEVEL% EQU 0 goto success

echo.
echo [2/3] Trying with O2 optimizations...
nvcc -O2 -o cuda.exe cuda.cu
if %ERRORLEVEL% EQU 0 goto success

echo.
echo [3/3] Trying with O2 and specific architecture...
nvcc -O2 -arch=sm_75 -o cuda.exe cuda.cu
if %ERRORLEVEL% EQU 0 goto success

echo.
echo All compilation attempts failed.
echo CUDA 13.0 may have compatibility issues.
echo Consider downgrading to CUDA 12.6.
pause
exit /b 1

:success
echo.
echo SUCCESS! cuda.exe compiled successfully
echo.
echo To run: cuda.exe weatherAUS.csv simple
echo Or: cuda.exe weatherAUS.csv tiled
pause

