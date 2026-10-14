# Make the Windows bundle: a zip of what a person needs to recompile a game of their own with a
# `cargo build --release --workspace` that is already done.
#
# usage: pwsh -File tools/bundle.ps1 [-Out <folder>] [-Selftest <folder>] [-NoSelftest]
#
# * -Out is where the staging folder and the zip go (default `dist-out`, which is ignored by git).
# * -Selftest is the folder that the ignored test `build::e2e_tests::writes_the_selftest_program`
#   wrote (`RECOMP_SELFTEST_FIXTURE=<folder> cargo test -p recomp --lib -- --ignored --exact
#   build::e2e_tests::writes_the_selftest_program`); its `exefs` goes in the bundle for
#   `selftest.cmd`. A bundle without one is refused unless -NoSelftest says so.
#
# The bundle holds recomp-gui.exe, recomp.exe and nx_runtime.lib from target\release, the README,
# doctor.cmd and selftest.cmd of `dist\`, and VERSION.txt. No game data, keys or firmware: nothing
# is read from anywhere but target\release, dist\ and -Selftest.
param(
    [string]$Out = "dist-out",
    [string]$Selftest = "",
    [switch]$NoSelftest
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$release = Join-Path $repo "target\release"
$files = @("recomp-gui.exe", "recomp.exe", "nx_runtime.lib")
foreach ($f in $files) {
    if (-not (Test-Path (Join-Path $release $f))) {
        throw "$f is not in ${release}: run cargo build --release --workspace first"
    }
}
if (-not $NoSelftest) {
    if ($Selftest -eq "" -or -not (Test-Path (Join-Path $Selftest "exefs\main"))) {
        throw "-Selftest must name the folder that holds exefs\main (see the usage), or pass -NoSelftest"
    }
}

# The version, the commit and the date.
$version = (Select-String -Path (Join-Path $repo "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' |
    Select-Object -First 1).Matches[0].Groups[1].Value
$sha = "unknown"
try {
    $s = (git rev-parse --short HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $s) { $sha = $s.Trim() }
} catch { }
$dirty = ""
try {
    if ((git status --porcelain 2>$null)) { $dirty = "+modified" }
} catch { }
$date = Get-Date -Format "yyyy-MM-dd"
$name = "recomp-$version-$sha-windows-x64"

$outDir = Join-Path $repo $Out
$stage = Join-Path $outDir $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

foreach ($f in $files) {
    Copy-Item (Join-Path $release $f) (Join-Path $stage $f)
}

# Text files go out with Windows line endings (a batch file with bare line feeds can lose its way
# to a label), with the version filled in.
function Copy-Text($from, $to) {
    $text = Get-Content -Raw -Path $from
    $text = $text.Replace("@VERSION@", $version).Replace("@SHA@", "$sha$dirty").Replace("@DATE@", $date)
    $text = ($text -replace "`r?`n", "`r`n")
    [System.IO.File]::WriteAllText($to, $text, (New-Object System.Text.ASCIIEncoding))
}
Copy-Text (Join-Path $repo "dist\README.txt") (Join-Path $stage "README.txt")
Copy-Text (Join-Path $repo "dist\doctor.cmd") (Join-Path $stage "doctor.cmd")
Copy-Text (Join-Path $repo "dist\selftest.cmd") (Join-Path $stage "selftest.cmd")

if (-not $NoSelftest) {
    $target = Join-Path $stage "selftest\exefs"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item (Join-Path $Selftest "exefs\*") $target -Recurse
}

$rustc = "unknown"
try { $rustc = (rustc --version 2>$null).Trim() } catch { }
$lines = @(
    "recomp $version",
    "commit $sha$dirty",
    "built $date with $rustc",
    "host $([System.Environment]::OSVersion.VersionString)"
)
[System.IO.File]::WriteAllText((Join-Path $stage "VERSION.txt"), (($lines -join "`r`n") + "`r`n"), (New-Object System.Text.ASCIIEncoding))

$zip = Join-Path $outDir "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal

Write-Host "bundle: $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
Get-ChildItem $stage -Recurse -File | ForEach-Object {
    "  {0,12:N0}  {1}" -f $_.Length, $_.FullName.Substring($stage.Length + 1)
} | Write-Host
