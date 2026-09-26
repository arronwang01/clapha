@echo off
rem il.train on the 4080; log in clapha\runs\<out>.log. Arguments: INIT OUT [extra il.train flags]
rem   train.cmd fl:hog2 train-hog26
rem   train.cmd fl:hog2 smoke --smoke
setlocal
set PY=D:\crtrain\py312\python.exe
cd /d %~dp0clapha
"%PY%" -m il.train --frames runs\conv-hog26 --init %1 --out runs\%2 --workers 14 %3 %4 %5 %6 > runs\%2.log 2>&1
