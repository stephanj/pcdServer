@echo off
rem Starts PCD Server from this folder and keeps the window open if it fails,
rem so the error stays readable. Extra arguments are passed through.
cd /d "%~dp0"
pcd_server.exe %*
if errorlevel 1 pause
