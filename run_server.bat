@echo off
chcp 65001 >nul
title Запуск PredSys

echo [1/2] Подготовка окружения...
call "%~dp0setup.bat"
if %errorlevel% neq 0 (
    echo Не удалось подготовить окружение. Код: %errorlevel%
    pause
    exit /b 1
)

echo [2/2] Запуск сервисов...
if not exist logs mkdir logs
start "Streamlit UI" cmd /k "call .venv\Scripts\activate.bat && streamlit run src\app.py"
set PYTHONPATH=%CD%\src&& uvicorn server:app --reload --host 0.0.0.0 --port 8000"

echo Нажмите любую клавишу для выхода...
pause >nul