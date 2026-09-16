@echo off
REM ===========================================================================
REM  DocMind 一键启动
REM
REM  逻辑同样放在 scripts\start.py 里, 原因见 install.bat 的说明.
REM ===========================================================================
chcp 65001 >nul 2>&1
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR] Python not found. / 未找到 python 命令
    echo   Please run install.bat after installing Python 3.11+.
    echo.
    pause
    exit /b 1
)

python "scripts\start.py" %*
set "EXITCODE=%errorlevel%"

if not "%EXITCODE%"=="0" (
    echo.
    echo   Exited with code %EXITCODE% / 服务异常退出
    echo.
    pause
)
exit /b %EXITCODE%
