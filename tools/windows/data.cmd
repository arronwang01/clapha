@echo off
rem Unpack every conv-hog26-*.zip next to this script into clapha\runs\conv-hog26 (the converted
rem replays; each zip holds new replay files and the index as of when it was made).
setlocal
cd /d %~dp0
for %%f in (conv-hog26-*.zip) do powershell -NoProfile -Command "Expand-Archive -Force '%%f' clapha" && move /y "%%f" done-%%f >nul
dir /s /b clapha\runs\conv-hog26\frames | find /c ".zst"
