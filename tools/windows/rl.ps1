# RL on the 4080 PC: il/rl_serve.py (every model's forward, GPU) + il/rl_learn.py (PPO, GPU) + one
# league collector per engine of the MuMu cluster (ports 26789.., CPU). Run from clapha-train:
#   powershell -ExecutionPolicy Bypass -File rl.ps1 -Run pilot1
#   powershell -ExecutionPolicy Bypass -File rl.ps1 -Run pilot1 -NoLearn        (collection only)
#   powershell -ExecutionPolicy Bypass -File rl.ps1 -Run pilot1 -Stop
# The learner starts from -Init (or the run's latest.pt when resuming) and writes latest.pt; the
# server serves `latest` to new games as it changes. Each game's opponent is drawn from -League
# (il/rl.py collect_remote): self, snap (past snapshots), anchor (-Init, fixed), hog2 (no delay),
# general (FirstLight's General on a real game's deck, no delay). Logs and PIDs in runs\rl\<Run>.
param(
    [Parameter(Mandatory)][string]$Run,
    [string]$Init = 'runs\distill-v2\checkpoint-00024489.pt',
    [int]$Engines = 8,
    [int]$Servers = 1,
    [int]$BasePort = 26789,
    [string]$League = 'self=4/snap=2/anchor=2/hog2=1/general=2',
    [string]$Ports = '',        # the engine ports to use, e.g. '26789/26790/26792' (default: -Engines from -BasePort)
    [string]$LearnArgs = '',
    [switch]$NoLearn,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$py = 'D:\crtrain\py312\python.exe'
Set-Location (Join-Path $PSScriptRoot 'clapha')
$dir = "runs\rl\$Run"
$pids = "$dir\pids.txt"
if ($Stop) {
    if (Test-Path $pids) {
        $ids = @(Get-Content $pids | ForEach-Object { [int]$_ })
        # only this run's processes: after a reboot an old id can belong to anything (the other user's job)
        Get-CimInstance Win32_Process | Where-Object { ($ids -contains $_.ProcessId) -and ($_.CommandLine -like "*rl\$Run*") } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Remove-Item $pids
    }
    'stopped'; exit 0
}
if (Test-Path $pids) { throw "$Run is running (pids in $pids); -Stop first" }
if (($League -match 'general') -and -not (Test-Path 'ref-firstlight\checkpoints\General\checkpoint-step-00000460.pt')) {
    throw 'FirstLight General checkpoint missing: ref-firstlight\checkpoints\General\checkpoint-step-00000460.pt'
}
New-Item -ItemType Directory -Force "$dir\games" | Out-Null
if (-not (Test-Path "$dir\latest.pt")) { Copy-Item $Init "$dir\latest.pt" }

function Launch($name, $arguments) {
    $p = Start-Process -FilePath $py -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$dir\$name.log" -RedirectStandardError "$dir\$name.err"
    Add-Content $pids $p.Id
    "$name pid $($p.Id)"
}
if (-not (Test-Path 'runs\rl\deals.pkl')) {
    'preparing the deal pool (once, ~12 min)...'
    & $py -m il.rl deals *> 'runs\rl\deals.log'
}
# several inference servers share the GPU (each one's Python work is its limit); collector i uses
# server i mod Servers
for ($k = 0; $k -lt $Servers; $k++) {
    Launch "serve-$k" "-m il.rl_serve --run $dir --device cuda --address 127.0.0.1:$(26900 + $k)"
}
if (-not $NoLearn) { Launch 'learn' "-m il.rl_learn --init $Init --games $dir\games --out $dir $LearnArgs" }
Start-Sleep 20      # the server loads its first models before the collectors ask
if ($Ports) { $portList = @($Ports -split '[/, ]+' | Where-Object { $_ } | ForEach-Object { [int]$_ }) }
else { $portList = @(0..($Engines - 1) | ForEach-Object { $BasePort + $_ }) }
# a fresh seed per start: the run is restarted often (every turn on the GPU) and a fixed seed would
# replay the same first deals each time
for ($i = 0; $i -lt $portList.Count; $i++) {
    $port = $portList[$i]
    Launch "collect-$port" "-m il.rl collect --server 127.0.0.1:$(26900 + $i % $Servers) --policy $dir\latest.pt --anchor $Init --league $League --games 1000000 --out $dir\games --port $port --seed $(Get-Random -Maximum 1000000000)"
}
