@echo off
rem Unpack the training code next to this script and check the GPU stack (run once, then after
rem each new clapha-train-code.zip). D:\crtrain\py312 is the Python with torch cu124 (the system
rem Python312 under AppData has no torch).
setlocal
set PY=D:\crtrain\py312\python.exe
cd /d %~dp0
powershell -NoProfile -Command "Expand-Archive -Force clapha-train-code.zip ."
"%PY%" -m pip install --quiet --disable-pip-version-check orjson zstandard
"%PY%" -c "import torch, orjson, zstandard; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
