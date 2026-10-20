param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& $Python (Join-Path $Root "src\run_revision_experiments.py") `
    --data $DataPath `
    --config (Join-Path $Root "configs\revision.json") `
    --out (Join-Path $Root "results_reproduced")

& $Python (Join-Path $Root "src\postprocess_results.py") `
    --results (Join-Path $Root "results_reproduced")
& $Python (Join-Path $Root "src\build_paper_tables.py") `
    --results (Join-Path $Root "results_reproduced")
& $Python (Join-Path $Root "src\make_revision_figures.py") `
    --results (Join-Path $Root "results_reproduced") `
    --out (Join-Path $Root "figures_reproduced")

Write-Host "Reproduction completed in results_reproduced and figures_reproduced."

