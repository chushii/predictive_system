@echo off
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install --quiet -r requirements.txt
py -m ipykernel install --user --name predsys --display-name "predsys"
exit /b %errorlevel%