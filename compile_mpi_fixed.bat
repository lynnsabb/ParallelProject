@echo off
REM Fixed MPI compilation script for MS-MPI

echo Setting up Visual Studio x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not set up Visual Studio environment
    echo Make sure Visual Studio 2022 Community is installed
    pause
    exit /b 1
)

echo.
echo Compiling MPI code...
cl /O2 /EHsc /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" mpi.c /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib /out:mpi.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! mpi.exe compiled successfully
    echo.
    echo To run: mpiexec -n 4 mpi.exe weatherAUS.csv
) else (
    echo.
    echo ERROR: Compilation failed
    echo Check error messages above
)

pause

