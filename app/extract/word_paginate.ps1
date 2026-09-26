# Word 本体に docx を組版させ、改ページ記録を書き込んだ複製を作る。
#
# app/extract/pagemap.py から呼ばれる。人手の GUI 操作ではなく、
# インジェストの一工程として起動する（LibreOffice を呼ぶのと同じ扱い）。
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass \
#     -File word_paginate.ps1 -Manifest <in.json> -Result <out.json>
#
# Manifest: [{"id": "...", "in": "C:\\...\\in.docx", "out": "C:\\...\\out.docx"}]
# Result  : [{"id": "...", "pages": 9, "error": ""}]
#
# 元ファイルは触らない（呼び出し側が一時ディレクトリへ複製済みのものを渡す）。
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Result
)

$ErrorActionPreference = 'Stop'
$wdFormatXMLDocument = 16
$wdDoNotSaveChanges  = 0
$wdStatisticPages    = 2
$wdAlertsNone        = 0
$msoAutomationSecurityForceDisable = 3

$jobs = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$out = New-Object System.Collections.ArrayList
$word = $null
$createdByUs = $false

try {
    $word = New-Object -ComObject Word.Application
    # 利用者が Word を開いている最中かもしれない。開いている文書が無いときだけ
    # 自分が起動したとみなし、最後に終了させる（他人の作業を閉じない）。
    $createdByUs = ($word.Documents.Count -eq 0)
    $word.Visible = $false
    $word.DisplayAlerts = $wdAlertsNone
    $word.AutomationSecurity = $msoAutomationSecurityForceDisable
    try { $word.Options.SaveInterval = 0 } catch {}
    try { $word.Options.UpdateLinksAtOpen = $false } catch {}

    foreach ($j in $jobs) {
        $rec = [ordered]@{ id = $j.id; pages = -1; error = '' }
        $doc = $null
        try {
            # ReadOnly:$false / AddToRecentFiles:$false / パスワードはダミーを渡し、
            # 保護文書でもダイアログを出さずに失敗させる
            $doc = $word.Documents.Open($j.in, $false, $false, $false, 'x', 'x', $false)
            $doc.Repaginate()
            $rec.pages = [int]$doc.ComputeStatistics($wdStatisticPages)
            try   { $doc.SaveAs2($j.out, $wdFormatXMLDocument) }
            catch { $doc.SaveAs($j.out, $wdFormatXMLDocument) }
        }
        catch {
            $rec.error = $_.Exception.Message
        }
        finally {
            if ($null -ne $doc) {
                try { $doc.Close($wdDoNotSaveChanges) } catch {}
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
            }
        }
        [void]$out.Add($rec)
    }
}
catch {
    [void]$out.Add([ordered]@{ id = '*'; pages = -1; error = $_.Exception.Message })
}
finally {
    if ($null -ne $word) {
        if ($createdByUs) { try { $word.Quit($wdDoNotSaveChanges) } catch {} }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}

$json = ConvertTo-Json -InputObject @($out) -Depth 4
Set-Content -LiteralPath $Result -Value $json -Encoding UTF8
