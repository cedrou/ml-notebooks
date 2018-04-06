@echo off

set CONDA_PATH=C:\Users\cedro\Anaconda3
set ENVNAME=tensorflow-gpu


call %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%
call activate %ENVNAME%
jupyter notebook

pause