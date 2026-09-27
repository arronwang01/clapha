@echo off
rem Start the unattended RL keeper (clapha\il\rl_turns.py) for a run (default pilot1): it takes turns on
rem the GPU with the other user's jobs, restarts whatever breaks (the engines and their VM included) and
rem rewrites clapha\runs\rl\<run>\status.txt every 10 minutes. Safe to run twice: one keeper per run.
rem Sharing (the user's call, 2026-09-27 evening): run next to the other job at full speed, back off while
rem other processes hold more than 7 GB of GPU memory (--others-gb 7; 1.5 = only when nobody trains).
rem 12 engines, not 16: next to the other job (15 GB + 9 GB of workers) the PC ran out of RAM (commit).
setlocal
set RUN=%1
if "%RUN%"=="" set RUN=pilot1
cd /d "%~dp0clapha"
if not exist runs\rl\%RUN% mkdir runs\rl\%RUN%
powershell -NoProfile -Command "Start-Process -FilePath 'D:\crtrain\py312\python.exe' -ArgumentList '-m','il.rl_turns','--run','%RUN%','--engines','12','--others-gb','7','--calm','2' -WorkingDirectory '%~dp0clapha' -WindowStyle Hidden"
echo keeper for %RUN% started. Status: %~dp0clapha\runs\rl\%RUN%\status.txt
