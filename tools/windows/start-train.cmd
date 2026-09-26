@echo off
rem Start train.cmd detached from this console (it keeps running when the ToDesk terminal closes).
cd /d %~dp0
if not exist clapha\runs mkdir clapha\runs
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath cmd.exe -ArgumentList '/c','%~dp0train.cmd %*'"
echo started: train.cmd %*   (log: clapha\runs\%2.log)
