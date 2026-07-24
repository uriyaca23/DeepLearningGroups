param(
    [Parameter(Mandatory = $true)]
    [string]$DocxPath,

    [string]$ContractPath = 'style/homework_style_contract.json',

    [string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python."
}

& $python (Join-Path $PSScriptRoot 'audit_homework_docx.py') `
    $DocxPath `
    --contract $ContractPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($PdfPath) {
    & (Join-Path $PSScriptRoot 'render_homework_word.ps1') `
        -DocxPath $DocxPath `
        -PdfPath $PdfPath
}
else {
    Write-Host 'Word render not run: pass -PdfPath to execute the Microsoft Word PDF gate.'
}
