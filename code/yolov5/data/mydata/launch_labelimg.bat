@echo off
cd /d "%~dp0"
set ROOT=%~dp0..\..\..\..\
set PY=%ROOT%.venv\Scripts\python.exe
if not exist "%PY%" (
  echo python not found: %PY%
  pause
  exit /b 1
)
echo.
echo === LabelImg for vehicle-countor ===
echo Open : %CD%\images
echo Save : %CD%\xml
echo Classes: person / car
echo.
echo IMPORTANT:
echo  1. Left list: click london_000.jpg (not 1.jpg)
echo  2. Press W, then drag a box on the image
echo  3. Choose person or car, Ctrl+S save, D next
echo  4. Zoom: Ctrl + mouse wheel, or toolbar + / -
echo.
"%PY%" "%~dp0run_labelimg.py"
if errorlevel 1 pause
