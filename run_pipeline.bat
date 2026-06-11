@echo off
:: =====================================================================
:: 🚗 SAFETRAVELS INDUSTRIAL ORCHESTRATION PIPELINE CONTROL HUB
:: =====================================================================
title SafeTravels ML Pipeline Engine Control Hub
setlocal enabledelayedexpansion

echo =====================================================================
echo 🛰️  STEP 1: Activating Isolated Environment Playground (venv)...
echo =====================================================================
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 🛑 CRITICAL ERROR: Failed to step inside python virtual environment.
    pause
    exit /b %errorlevel%
)
echo ✓ Virtual environment loaded successfully.
echo.

:menu
echo =====================================================================
echo 🧭 CONTROL ROOM: Select an Operations Phase to Trigger
echo =====================================================================
echo [1] Run Data Engineering Pipeline (Harvest Weather & Build Matrices)
echo [2] Execute Machine Learning Tournament (Train Model & Lock Schemas)
echo [3] Launch Complete Production Network (FastAPI Backend + Streamlit UI)
echo [4] Execute Automated Integration Test Suite (Pytest Framework)
echo [5] Exit Control Room
echo.

set /p choice="Enter choice index number (1-5): "

if "%choice%"=="1" goto data_prep
if "%choice%"=="2" goto train_model
if "%choice%"=="3" goto launch_app
if "%choice%"=="4" goto run_tests
if "%choice%"=="5" goto end
echo ❌ Invalid selection profile indicator. Please choose a target between 1 and 5.
echo.
goto menu

:data_prep
echo.
echo =====================================================================
echo 📊 PHASE 01: Initializing End-to-End Data Generation & Cleansing...
echo =====================================================================
python src/pipeline_data_prep.py
if %errorlevel% neq 0 (
    echo ⚠ Pipeline compilation disrupted with non-zero processing delta logs.
) else (
    echo ✓ Data extraction matrix layers locked successfully!
)
echo.
pause
goto menu

:train_model
echo.
echo =====================================================================
echo 🥇 PHASE 02: Initializing ML Model Tournament Validation Engine...
echo =====================================================================
python src/pipeline_train_model.py
if %errorlevel% neq 0 (
    echo ⚠ Model training pipeline encountered a validation compilation fault.
) else (
    echo ✓ Champion model weights binary frozen inside models directory!
)
echo.
pause
goto menu

:launch_app
echo.
echo =====================================================================
echo 🚀 PHASE 03: Launching Decoupled Enterprise Software Clusters...
echo =====================================================================
echo 🔌 Deploying FastAPI server layer on port 8000...
:: Using unique titles so our tracking engine can gracefully reference them later
start "ST_BACKEND" /min cmd /c "call venv\Scripts\activate.bat && python backend/main.py"

echo 🎨 Deploying UI Streamlit visualization engine tier...
start "ST_FRONTEND" /min cmd /c "call venv\Scripts\activate.bat && streamlit run frontend/app_streamlit.py"

echo.
echo 🟢 SYSTEM STATE: RUNNING
echo ---------------------------------------------------------------------
echo Core clusters are actively processing transactions in the background.
echo [!] CRITICAL: To spin down servers and free up network ports, 
echo     press ANY KEY inside this window to run the teardown cycle.
echo ---------------------------------------------------------------------
echo.
pause

echo.
echo =====================================================================
echo 🛑 SHUTDOWN INITIATED: Cleanly Killing Background Server Processes...
echo =====================================================================
:: Professionally target and terminate only our project windows via their process titles
taskkill /FI "WINDOWTITLE eq ST_BACKEND*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ST_FRONTEND*" /T /F >nul 2>&1
echo ✓ Port 8000 and Port 8501 cluster environments recycled successfully.
echo.
pause
goto menu

:run_tests
echo.
echo =====================================================================
echo 🧪 PHASE 04: Running Automated Pydantic & API Endpoint Assertions...
echo =====================================================================
pytest tests/
echo.
pause
goto menu

:end
echo.
echo Closing Control Room Session Hub. Have a great run, developer!
timeout /t 3 >nul
exit /b 0