@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-job.ps1" %*
