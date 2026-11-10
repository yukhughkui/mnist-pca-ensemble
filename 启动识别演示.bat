@echo off
cd /d "%~dp0"
set "PY=C:\Users\28215\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist "%PY%" set "PY=python"
echo.
echo   正在启动手写数字识别演示系统 ...
echo   浏览器会自动打开；若未打开，请手动访问 http://127.0.0.1:8765
echo   想停止服务，直接关闭本窗口即可。
echo.
"%PY%" -u src\web_demo.py
if errorlevel 1 (
  echo.
  echo   启动失败：请确认依赖已安装（pip install -r requirements.txt）
  pause
)
