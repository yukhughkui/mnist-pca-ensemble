@echo off
cd /d "%~dp0"
set "PY=C:\Users\28215\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist "%PY%" set "PY=python"
echo.
echo   正在用最新实验数据重新生成论文 docx ...
echo.
"%PY%" -c "import sys,json,pathlib; d=json.loads(pathlib.Path('results/metrics.json').read_text(encoding='utf-8')); sys.exit(0 if 'confusion' in d else 1)"
if errorlevel 1 (
  echo   指标文件缺少混淆矩阵，正在自动补齐 ...
  "%PY%" -u src\fix_confusion.py
)
"%PY%" report\make_reference.py
"%PY%" -u report\make_paper.py
echo.
echo   完成。论文在 report\ 目录下。
pause
