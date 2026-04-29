param(
    [string]$SubDir = ""
)

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
$RemotePath = "workspace/$Username/$folderName"

# 以 mutagen.config.yml 为模板，替换占位符，生成 mutagen.yml
$content = Get-Content -Path "$PSScriptRoot\mutagen.config.yml" -Raw
$content = $content -replace '\{WORK_DIR\}', $RemotePath
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
$RemoteCmdWorkDir = $RemoteWorkDir -replace '/', '\\'
Write-Host "已生成 mutagen.yml，同步路径: $RemotePath，工作目录: $RemoteWorkDir"
Write-Host "执行: ssh vibecoding-server -> cd /d $RemoteCmdWorkDir -> claude"
ssh -t vibecoding-server "cd /d `"$RemoteCmdWorkDir`" && claude"

# SSH 退出后，显示 mutagen 同步状态
Write-Host ""
Write-Host "已退出远端服务器，查看 Mutagen 状态..." -ForegroundColor Yellow
mutagen project list --long
