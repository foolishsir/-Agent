@echo off
REM ===========================================================================
REM  DocMind 后端一键启动 (Windows)
REM  用法: 双击本文件, 或在本目录执行 run-backend.cmd
REM ===========================================================================
setlocal

cd /d "%~dp0backend"

if not exist "..\.env" (
    echo [WARN] 未找到 .env, 正在从 .env.example 复制...
    copy "..\.env.example" "..\.env" >nul
    echo [WARN] 请编辑 .env 填入 DOCMIND_LLM_API_KEY 后重新运行.
)

echo.
echo ============================================================
echo  DocMind API  ->  http://127.0.0.1:8000
echo  接口文档      ->  http://127.0.0.1:8000/docs
echo  就绪探针      ->  http://127.0.0.1:8000/api/v1/health/ready
echo ============================================================
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

endlocal
