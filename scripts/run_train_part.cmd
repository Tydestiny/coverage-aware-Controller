@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: scripts\run_train_part.cmd task_ids\train_parts\train_task_ids_part_1.txt
  exit /b 2
)

set "CONTROLLER_REPO=%~dp0.."
for %%I in ("%CONTROLLER_REPO%") do set "CONTROLLER_REPO=%%~fI"

if "%APPWORLD_ROOT%"=="" (
  set "APPWORLD_ROOT=D:\TY\SJTU\CIS\adaptive_compaction_validation\appworld"
)

set "TASK_LIST=%~f1"
set "UV_CACHE_DIR=%CONTROLLER_REPO%\.uv-cache"

echo Controller repo: %CONTROLLER_REPO%
echo AppWorld root:   %APPWORLD_ROOT%
echo Task list:       %TASK_LIST%
echo.

cd /d "%APPWORLD_ROOT%" || exit /b 1

for /f "usebackq tokens=*" %%t in ("%TASK_LIST%") do (
  if not "%%t"=="" (
    echo === Collecting task %%t ===
    uv run python "%CONTROLLER_REPO%\scripts\run_controller_api_smoke.py" --execute --task-id %%t --capture-checkpoints --max-cost-cny 1.0 --appworld-root "%APPWORLD_ROOT%"
    if errorlevel 1 (
      echo TASK_FAILED %%t
    )
  )
)

endlocal
