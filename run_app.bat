@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%langchain-env\Scripts\python.exe"
set "APP_PY=%SCRIPT_DIR%app.py"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found at "%PYTHON_EXE%"
    exit /b 1
)

if not exist "%APP_PY%" (
    echo app.py not found at "%APP_PY%"
    exit /b 1
)

"%PYTHON_EXE%" "%APP_PY%" %*
