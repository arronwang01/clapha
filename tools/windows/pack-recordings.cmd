@echo off
rem The training games every run kept since the last pack, in ONE zip for ToDesk: clapha-recordings.zip
rem next to this file. On the Mac: Clapha -> Training games -> Import, and pick that zip (or drop it on
rem the window). "--models" also brings each run's newest model (~50 MB each) for playing it live in the
rem console; "--all" packs every game again (a lost zip).
setlocal
cd /d "%~dp0clapha"
D:\crtrain\py312\python.exe -m il.recordings pack --out "%~dp0clapha-recordings.zip" %*
