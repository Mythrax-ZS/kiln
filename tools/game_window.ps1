# Look at and drive the window of a produced program without touching the rest of the desktop.
#
# usage: pwsh -File tools/game_window.ps1 -ProcessId <pid> [-Capture <png>] [-Keys <list>] [-Wait <ms>]
#
# * -Capture writes the picture of the program's own window (its client area, as the window shows
#   it: `PrintWindow`, never a screenshot of the screen) to the png.
# * -Keys is a comma separated list of `<key>:<milliseconds>` entries, pressed one after the other:
#   the key is a name of the table below (the physical keys the standard keymap uses, or F13 to
#   F24, which nobody types: run the program with an `NX_KEYMAP` file that binds those, so that
#   the person at the keyboard cannot disturb the run). The keys go to the window's own message
#   queue with `PostMessage`, not to whichever window has the focus.
# * -Wait is a pause in milliseconds before the keys and the capture.
#
# It prints the size of the window's client area. Delete the program's key traces (`NX_TRACE_KEYS`)
# afterwards: they hold what was pressed.
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [string]$Capture = "",
    [string]$Keys = "",
    [int]$Wait = 0
)

Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class GameWindow {
    delegate bool EnumProc(IntPtr hwnd, IntPtr lparam);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc proc, IntPtr lparam);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool GetClientRect(IntPtr hwnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wparam, IntPtr lparam);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint type);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    // The visible top-level windows of the process, the biggest first.
    public static IntPtr Find(uint pid) {
        var found = new List<IntPtr>();
        EnumWindows((h, l) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p == pid && IsWindowVisible(h)) found.Add(h);
            return true;
        }, IntPtr.Zero);
        IntPtr best = IntPtr.Zero; long area = -1;
        foreach (var h in found) {
            RECT r; GetClientRect(h, out r);
            long a = (long)(r.Right - r.Left) * (r.Bottom - r.Top);
            if (a > area) { area = a; best = h; }
        }
        return best;
    }
    public static int[] Size(IntPtr hwnd) {
        RECT r; GetClientRect(hwnd, out r);
        return new int[] { r.Right - r.Left, r.Bottom - r.Top };
    }
}
"@
Add-Type -AssemblyName System.Drawing

# Without this, on a scaled display the client rectangle and what PrintWindow draws disagree and the
# capture is only the top-left part of the window.
[void][GameWindow]::SetProcessDPIAware()

$hwnd = [GameWindow]::Find([uint32]$ProcessId)
if ($hwnd -eq [IntPtr]::Zero) {
    Write-Error "no visible window of process $ProcessId"
    exit 1
}
$size = [GameWindow]::Size($hwnd)
Write-Output "window client area $($size[0]) x $($size[1])"
if ($Wait -gt 0) { Start-Sleep -Milliseconds $Wait }

# Virtual key codes of the keys a keymap names (the letters and digits are their codes).
$vk = @{
    "Enter" = 0x0D; "Backspace" = 0x08; "Space" = 0x20; "Escape" = 0x1B; "Tab" = 0x09
    "ShiftLeft" = 0xA0; "ShiftRight" = 0xA1
    "ArrowLeft" = 0x25; "ArrowUp" = 0x26; "ArrowRight" = 0x27; "ArrowDown" = 0x28
}
foreach ($c in [char[]]"ABCDEFGHIJKLMNOPQRSTUVWXYZ") { $vk["Key$c"] = [int]$c }
foreach ($n in 13..24) { $vk["F$n"] = 0x70 + $n - 1 }
$extended = @("ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown")

function Send-Key([string]$name, [int]$down) {
    if (-not $vk.ContainsKey($name)) { Write-Error "unknown key $name"; return }
    $code = [uint32]$vk[$name]
    $scan = [GameWindow]::MapVirtualKey($code, 0)
    $flags = 1 -bor ($scan -shl 16)
    if ($extended -contains $name) { $flags = $flags -bor (1 -shl 24) }
    [void][GameWindow]::PostMessage($hwnd, 0x0100, [IntPtr]$code, [IntPtr]$flags)
    Start-Sleep -Milliseconds $down
    $flags = $flags -bor (1 -shl 30) -bor (1 -shl 31)
    [void][GameWindow]::PostMessage($hwnd, 0x0101, [IntPtr]$code, [IntPtr]$flags)
}

foreach ($entry in ($Keys -split "," | Where-Object { $_ -ne "" })) {
    $name, $ms = $entry.Split(":")
    Send-Key $name ([int]$ms)
}

if ($Capture -ne "") {
    $bitmap = New-Object System.Drawing.Bitmap($size[0], $size[1])
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $hdc = $g.GetHdc()
    [void][GameWindow]::PrintWindow($hwnd, $hdc, 3)
    $g.ReleaseHdc($hdc)
    $g.Dispose()
    $bitmap.Save($Capture, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
    Write-Output "wrote $Capture"
}
