@echo off
setlocal
title Start Applio

:::
:::                       _ _
:::     /\               | (_)
:::    /  \   _ __  _ __ | |_  ___
:::   / /\ \ | '_ \| '_ \| | |/ _ \
:::  / ____ \| |_) | |_) | | | (_) |
::: /_/    \_\ .__/| .__/|_|_|\___/
:::          | |   | |
:::          |_|   |_|
:::
:::

:menu
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A

echo [1] Start Applio
echo [2] Start Applio (DML)
echo [3] Start Realtime GUI (DML)
echo [4] Start Realtime GUI (V0)
echo [5] Start Realtime GUI (V1)
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="1" (
    cls
    echo WARNING: At this point, it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
    echo.
    python infer-web.py --pycmd python --port 7897
    pause
    cls
    goto menu
)

if "%choice%"=="2" (
    cls
    echo.
    python infer-web.py --pycmd python --port 7897 --dml
    pause
    cls
    goto menu
)

if "%choice%"=="3" (
    cls
    echo Starting Realtime GUI ^(DML^)...
    echo.
    python tools/gui/gui_v1.py --pycmd python --dml
    pause
    cls
    goto menu
)

if "%choice%"=="4" (
    cls
    echo Starting Realtime GUI ^(V0^)...
    echo.
    python tools/gui/gui_v0.py
    pause
    cls
    goto menu
)

if "%choice%"=="5" (
    cls
    echo Starting Realtime GUI ^(V1^)...
    echo.
    python tools\gui\gui_v1.py
    pause
    cls
    goto menu
)

cls
echo Invalid option. Please enter a number from 1 to 5.
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
