@echo off
REM ===========================================================================
REM  DocMind 一键安装
REM
REM  这个文件只做一件事: 用当前的 Python 解释器运行 scripts\setup.py
REM
REM  为什么不把逻辑写在这里: cmd.exe 对中文的支持不可靠 —— 即使加了
REM  chcp 65001, 中文串仍可能被逐字节解析导致行被截断, 报出
REM  "'失败]' is not recognized as an internal or external command" 这种错.
REM  所以批处理只保留 ASCII 启动器, 所有逻辑与中文提示都放在 Python 脚本里.
REM ===========================================================================
chcp 65001 >nul 2>&1
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR] Python not found. / 未找到 python 命令
    echo.
    echo   Please install Python 3.11+ from:
    echo     https://www.python.org/downloads/
    echo.
    echo   Remember to check "Add Python to PATH" during installation.
    echo   安装时请务必勾选 "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

python "scripts\setup.py"
set "EXITCODE=%errorlevel%"

echo.
pause
exit /b %EXITCODE%
