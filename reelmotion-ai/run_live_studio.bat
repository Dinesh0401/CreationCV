@echo off
title ReelMotion AI - Live Character Studio
echo ========================================================
echo  ReelMotion AI - Live Character Replacement Studio
echo ========================================================
echo Starting camera and neural vision tracking...
cd /d "%~dp0"
set PYTHONPATH=%~dp0
.venv\Scripts\python.exe scripts\live_camera_studio.py
pause

