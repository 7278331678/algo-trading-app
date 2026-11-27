@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\backfill.ps1" %*
pause
