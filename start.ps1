param(
    [string]$SubDir = ""
)

function ConvertTo-RemoteShellLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + ($Value -replace "'", "'\''") + "'"
}

function ConvertTo-SafeName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $safeName = $Value -replace '[^A-Za-z0-9]', '-'
    $safeName = $safeName -replace '-+', '-'
    $safeName = $safeName.Trim('-')
    if ([string]::IsNullOrWhiteSpace($safeName)) {
        return "sync"
    }
    if ($safeName -match '^[0-9]') {
        $safeName = "sync-$safeName"
    }

    return $safeName
}

function Select-RemoteCommand {
    while ($true) {
        Write-Host "请选择进入的工具："
        Write-Host "1. Codex"
        Write-Host "2. Claude（默认）"
        $inputBuffer = Read-Host "请输入 [1/2]"

        switch ($inputBuffer.Trim().ToLowerInvariant()) {
            "1" { return "codex" }
            "codex" { return "codex" }
            "x" { return "codex" }
            "" { return "claude" }
            "2" { return "claude" }
            "claude" { return "claude" }
            "c" { return "claude" }
            default {
                Write-Host "输入无效，请输入 1 或 2" -ForegroundColor Yellow
            }
        }
    }
}

$RemoteCommand = Select-RemoteCommand

$configDir = "$env:USERPROFILE\.szjson"
$configFile = "$configDir\config.json"
$defaultUsername = ""
if (Test-Path $configFile) {
    $config = Get-Content $configFile -Raw | ConvertFrom-Json
    $defaultUsername = $config.vibecoding_username
}

if ($defaultUsername) {
    $Username = Read-Host "请输入账号名 [$defaultUsername]"
    if ([string]::IsNullOrWhiteSpace($Username)) {
        $Username = $defaultUsername
    }
} else {
    $Username = Read-Host "请输入账号名"
    if ([string]::IsNullOrWhiteSpace($Username)) {
        Write-Host "账号名不能为空" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $configDir)) { New-Item -ItemType Directory -Path $configDir | Out-Null }
$config = if (Test-Path $configFile) { Get-Content $configFile -Raw | ConvertFrom-Json } else { [PSCustomObject]@{} }
$config | Add-Member -MemberType NoteProperty -Name vibecoding_username -Value $Username -Force
$config | ConvertTo-Json | Set-Content -Path $configFile -NoNewline

# 取当前脚本所在目录的叶子名作为远端工作目录名
$folderName = Split-Path -Leaf $PSScriptRoot
$syncName = ConvertTo-SafeName $folderName
$RemotePath = "workspace/$Username/$folderName"

# 先确保远端目录存在。mutagen 可以同步内容，但 SSH 后续 cd 需要目录已经可进入。
$RemotePathLiteral = ConvertTo-RemoteShellLiteral $RemotePath
Write-Host "执行: ssh vibecoding -> mkdir -p -- $RemotePath"
ssh vibecoding "mkdir -p -- $RemotePathLiteral"
if ($LASTEXITCODE -ne 0) {
    Write-Host "创建远端目录失败: $RemotePath" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 以 mutagen.config.yml 为模板，替换占位符，生成 mutagen.yml
$content = Get-Content -Path "$PSScriptRoot\mutagen.config.yml" -Raw
$content = $content -replace '\{WORK_DIR\}', $RemotePath
$content = $content -replace '\{SYNC_NAME\}', $syncName
$content = $content -replace '\{FOLDER_NAME\}', $folderName
Set-Content -Path "$PSScriptRoot\mutagen.yml" -Value $content -NoNewline

# 先终止已有的 mutagen project，再启动
mutagen project terminate 2>$null
Write-Host "执行: mutagen project start"
mutagen project start
Write-Host "Mutagen 同步已启动，等待同步完成..."

# 等待同步状态就绪
do {
    Start-Sleep -Seconds 2
    $output = mutagen project list --long 2>&1 | Out-String
    # 提取 Status 行
    $statusLine = ($output -split "`n" | Where-Object { $_ -match '^\s*Status:' }) -join ''
    $statusLine = $statusLine.Trim()
    $time = Get-Date -Format "HH:mm:ss"
    # 原行覆盖输出
    Write-Host "`r[$time] $statusLine" -NoNewline
} while ($output -notmatch "Watching for changes")
Write-Host ""
Write-Host "同步完成" -ForegroundColor Green

# SSH 到远端服务器并进入目录
$RemoteWorkDir = $RemotePath
$TmuxSessionName = "$Username`_$folderName`_$(Get-Date -Format 'yyyyMMddHHmmss')"
$TmuxSessionName = $TmuxSessionName -replace '[^A-Za-z0-9_.-]', '_'
if (-not [string]::IsNullOrWhiteSpace($SubDir)) {
    # 规范化：保持同步配置用 /，去掉开头的 ./ 或 .\，去掉末尾分隔符
    $SubDir = $SubDir -replace '\\', '/'
    $SubDir = $SubDir -replace '^\./', ''
    $SubDir = $SubDir -replace '^\.\\', ''
    $SubDir = $SubDir.Trim('/')
    if (-not [string]::IsNullOrWhiteSpace($SubDir)) {
        $RemoteWorkDir = "$RemoteWorkDir/$SubDir"
    }
}

Write-Host "已生成 mutagen.yml，同步路径: $RemotePath，工作目录: $RemoteWorkDir"
Write-Host "执行: ssh vibecoding -> cd -- $RemoteWorkDir -> tmux $TmuxSessionName -> $RemoteCommand"
$RemoteWorkDirLiteral = ConvertTo-RemoteShellLiteral $RemoteWorkDir
$TmuxSessionNameLiteral = ConvertTo-RemoteShellLiteral $TmuxSessionName
$RemoteCommandLiteral = ConvertTo-RemoteShellLiteral "exec $RemoteCommand"
ssh -t vibecoding "cd -- $RemoteWorkDirLiteral && tmux new-session -s $TmuxSessionNameLiteral bash -lc $RemoteCommandLiteral"

# SSH 退出后，显示 mutagen 同步状态
Write-Host ""
Write-Host "已退出远端服务器，查看 Mutagen 状态..." -ForegroundColor Yellow
mutagen project list --long
