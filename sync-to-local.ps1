$dst = 'E:\Opensource\TTS\VoxCPM-TTS-Tool'
$src = 'vibecoding-server:workspace/wys/VoxCPM-TTS-Tool'

@('src\voxcpm_tts_tool\chunking.py',
  'src\voxcpm_tts_tool\generation_queue.py',
  'tests\test_chunking.py',
  'tests\test_generation_queue.py') | ForEach-Object {
    $p = Join-Path $dst $_
    if (Test-Path $p) { Remove-Item $p -Force; Write-Host "deleted $_" }
}

@('app.py',
  'src/voxcpm_tts_tool/app_state.py',
  'src/voxcpm_tts_tool/generation.py',
  'src/voxcpm_tts_tool/script_parser.py',
  'tests/test_app_state.py',
  'tests/test_generation.py',
  'tests/test_script_parser.py') | ForEach-Object {
    $local = Join-Path $dst ($_.Replace('/', '\'))
    $remote = "${src}/$_"
    & scp $remote $local
    Write-Host "fetched $_"
}

@('docs/superpowers/specs/2026-04-27-flatten-generation-tab-design.md',
  'docs/superpowers/plans/2026-04-27-flatten-generation-tab-impl.md') | ForEach-Object {
    $local = Join-Path $dst ($_.Replace('/', '\'))
    New-Item -ItemType Directory -Path (Split-Path $local) -Force | Out-Null
    & scp "${src}/$_" $local
    Write-Host "fetched $_"
}

Get-ChildItem $dst -Recurse -Filter '__pycache__' -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Write-Host "done."
