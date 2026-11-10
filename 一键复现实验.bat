@echo off
cd /d "%~dp0"
set "PY=C:\Users\28215\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist "%PY%" set "PY=python"
echo.
echo   正在复现全部实验（CPU 约 6 分钟），请耐心等待 ...
echo.
"%PY%" -u src\run_experiment.py
"%PY%" -u src\fix_confusion.py
"%PY%" -u src\compare_ens_mlp.py
"%PY%" -u src\calibration_check.py
"%PY%" -u src\application_demo.py
echo.
echo   完成。图表在 figures\，指标在 results\。
pause
