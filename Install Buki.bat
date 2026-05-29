@echo off
:: Buki Installer — double-click to run
:: Launches the PowerShell installer with a GUI window.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer.ps1"
