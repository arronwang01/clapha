@echo off
rem Stop the RL keeper and its run (default pilot1). Nothing of ours touches the GPU until
rem start-training.cmd is run again. The run continues from where it was (latest.pt).
setlocal
set RUN=%1
if "%RUN%"=="" set RUN=pilot1
cd /d "%~dp0"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*il.rl_turns*' -and $_.CommandLine -like '*%RUN%*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'keeper ' + $_.ProcessId + ' stopped' }"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rl.ps1" -Run %RUN% -Stop
