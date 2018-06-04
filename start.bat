@echo off

set CONDA_PATH=C:\Users\cedro\Anaconda3
set ENVNAME=tf


call %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%
call activate %ENVNAME%
jupyter lab

pause