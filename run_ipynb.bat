@echo off
chcp 65001 >nul
title Jupyter Notebook

echo [1/2] Подготовка окружения...
call "%~dp0setup.bat"
if %errorlevel% neq 0 (
    echo Не удалось подготовить окружение. Код: %errorlevel%
    pause
    exit /b 1
)

echo [2/2] Запуск Jupyter Notebook...
py -3.14 -m notebook