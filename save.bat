@echo off
if "%~1"=="" (
  echo Usage: save "type(TAG): message"
  exit /b 1
)
git add -A
git commit -m "%~1" || exit /b 1
git push