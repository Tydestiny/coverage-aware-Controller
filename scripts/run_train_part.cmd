@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: scripts\run_train_part.cmd task_ids\train_parts\train_task_ids_part_1.txt
  exit /b 2
)

set "CONTROLLER_REPO=%~dp0.."
for %%I in ("%CONTROLLER_REPO%") do set "CONTROLLER_REPO=%%~fI"

set "APPWORLD_ROOT_CHECK=%APPWORLD_ROOT: =%"
if "%APPWORLD_ROOT_CHECK%"=="" (
  set "APPWORLD_ROOT=!CONTROLLER_REPO!\..\appworld"
  for %%I in ("!APPWORLD_ROOT!") do set "APPWORLD_ROOT=%%~fI"
)
if "!APPWORLD_ROOT!"=="" (
  set "APPWORLD_ROOT=!CONTROLLER_REPO!\..\appworld"
  for %%I in ("!APPWORLD_ROOT!") do set "APPWORLD_ROOT=%%~fI"
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
    set "TASK_OUTPUT=!APPWORLD_ROOT!\experiments\outputs\controlled_react_checkpoint_collection\tasks\%%t"
    set "TASK_COMPLETED=0"
    if exist "!TASK_OUTPUT!\checkpoints\manifest.json" if exist "!TASK_OUTPUT!\collection\baseline_collection_report.md" set "TASK_COMPLETED=1"
    if "!TASK_COMPLETED!"=="1" (
      echo === Skipping completed task %%t ===
    ) else (
      echo === Collecting task %%t ===
      uv run python "%CONTROLLER_REPO%\scripts\collect_with_checkpoints.py" --execute --task-id %%t --max-cost-cny 1.0 --appworld-root "!APPWORLD_ROOT!"
      if errorlevel 1 (
        echo TASK_FAILED %%t
      )
    )
  )
)

endlocal
