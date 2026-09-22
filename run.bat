@echo off
title Custom Generative AI Studio
cd /d "%~dp0"

echo ========================================================
echo        CUSTOM GENERATIVE AI STUDIO (V-TON ^& INPAINT)
echo ========================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Menyiapkan virtual environment Python...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Memeriksa dependensi...
python -c "import torch" 2>nul
if %errorlevel% neq 0 (
    echo Menginstal PyTorch dengan dukungan CUDA...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
)

echo.
echo Menjalankan aplikasi studio...
python app.py

pause
