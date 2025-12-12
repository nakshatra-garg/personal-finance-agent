@echo off

echo Personal Finance Agent
echo =====================
echo.

REM Check if .env exists
if not exist .env (
    echo Warning: .env file not found!
    echo Creating .env from .env.example...
    copy .env.example .env
    echo.
    echo Created .env file
    echo Please edit .env and add your API keys before running the app
    echo.
    pause
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo Starting Streamlit app...
echo.

REM Run the app
streamlit run app.py
