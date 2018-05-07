@echo off

set CONDA_PATH=C:\Users\cedro\Anaconda2
set ENVNAME=tf-gpu


call %CONDA_PATH%\Scripts\activate.bat %ENVNAME%
jupyter lab

pause