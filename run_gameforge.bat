@echo off
setlocal
echo ========================================================
echo GameForge AI - Unified Production Server
echo ========================================================
cd /d "%~dp0"

if not exist "gui\dist\index.html" (
    echo [WARNING] gui\dist\index.html not found!
    echo Serving backend API mode...
)

echo Starting GameForge on http://localhost:8000 ...
timeout /t 2 >nul
start http://localhost:8000

backend\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause

