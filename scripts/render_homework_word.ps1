param(
    [Parameter(Mandatory = $true)]
    [string]$DocxPath,

    [Parameter(Mandatory = $true)]
    [string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$resolvedDocx = (Resolve-Path -LiteralPath $DocxPath).Path
$resolvedPdf = [System.IO.Path]::GetFullPath($PdfPath)
$pdfDirectory = Split-Path -Parent $resolvedPdf

if ([System.IO.Path]::GetExtension($resolvedDocx) -ne '.docx') {
    throw "Expected a .docx input: $resolvedDocx"
}

if (-not (Test-Path -LiteralPath $pdfDirectory)) {
    New-Item -ItemType Directory -Force -Path $pdfDirectory | Out-Null
}

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    # Read-only: this exports the exact saved candidate and cannot silently
    # mutate the DOCX while rendering it.
    $document = $word.Documents.Open($resolvedDocx, $false, $true)
    $wdExportFormatPDF = 17
    $document.ExportAsFixedFormat($resolvedPdf, $wdExportFormatPDF)

    [pscustomobject]@{
        DocxPath = $resolvedDocx
        PdfPath = $resolvedPdf
        Paragraphs = [int]$document.Paragraphs.Count
        TablesOfContents = [int]$document.TablesOfContents.Count
        ReadOnly = $true
    } | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
    if ($null -ne $document) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
