@echo off

if "%CONDA_PATH%"=="" (
    set CONDA_PATH=C:\Users\cedro\Anaconda2
)
set ENVNAME=musread


call %CONDA_PATH%\Scripts\activate.bat %ENVNAME%
jupyter lab

pause