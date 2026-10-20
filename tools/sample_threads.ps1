# What is a produced Windows program busy with? Samples the instruction pointer of every thread now and
# then and names where it is: a function of the program (from the link map that `kiln build` or
# `kiln relink` writes when KILN_LINK_MAP=<file> is set: generated code is `f_<guest address>`, the
# runtime's functions have their Rust names) or a module of the process (a driver, ntdll).
#
# usage: pwsh -File tools/sample_threads.ps1 -Map <map file> [-Process title] [-Samples 40]
#            [-IntervalMs 50] [-Top 25] [-OnlyRunning] [-Callers] [-Depth 1]
#
# -OnlyRunning leaves out the threads the system says are waiting. The first column is how many
# samples found a thread at the place, then the thread's id, then the place. -Callers adds, to a
# sample that is outside the program (a system library), the first place of the program that a word
# near the top of the thread's stack points into: usually who called it; -Depth 3 adds the next
# two such places, a rough backtrace (return addresses and spilled pointers alike).
param(
    [Parameter(Mandatory = $true)][string]$Map,
    [string]$Process = "title",
    [int]$Samples = 40,
    [int]$IntervalMs = 50,
    [int]$Top = 25,
    [switch]$OnlyRunning,
    [switch]$Callers,
    [int]$Depth = 1
)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ThreadIp {
    [DllImport("kernel32.dll")] static extern IntPtr OpenThread(uint access, bool inherit, uint tid);
    [DllImport("kernel32.dll")] static extern uint SuspendThread(IntPtr h);
    [DllImport("kernel32.dll")] static extern int ResumeThread(IntPtr h);
    [DllImport("kernel32.dll")] static extern bool GetThreadContext(IntPtr h, IntPtr ctx);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll")] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll")] static extern bool ReadProcessMemory(IntPtr h, IntPtr addr, byte[] buf, IntPtr size, out IntPtr read);
    static IntPtr process = IntPtr.Zero;
    // The words at the top of a thread's stack, as of the last call of Rip (null when there are none).
    public static ulong[] Stack;
    // The instruction pointer of a thread of this (64-bit) process, or 0 when it cannot be read.
    // With words > 0 the words at the top of its stack are left in Stack.
    public static ulong Rip(uint tid, uint pid, int words) {
        Stack = null;
        // THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION
        IntPtr h = OpenThread(0x0002 | 0x0008 | 0x0040, false, tid);
        if (h == IntPtr.Zero) return 0;
        // A CONTEXT is 1232 bytes, aligned to 16; ContextFlags (at 0x30) asks for the control registers
        // (CONTEXT_AMD64 | CONTEXT_CONTROL), and Rip is at 0xF8.
        IntPtr raw = Marshal.AllocHGlobal(1232 + 16);
        long aligned = ((long)raw + 15) & ~15L;
        IntPtr ctx = (IntPtr)aligned;
        for (int i = 0; i < 1232; i++) Marshal.WriteByte(ctx, i, 0);
        Marshal.WriteInt32(ctx, 0x30, 0x100001);
        ulong rip = 0;
        if (SuspendThread(h) != 0xFFFFFFFF) {
            if (GetThreadContext(h, ctx)) {
                rip = (ulong)Marshal.ReadInt64(ctx, 0xF8);
                if (words > 0) {
                    ulong rsp = (ulong)Marshal.ReadInt64(ctx, 0x98);
                    if (process == IntPtr.Zero) process = OpenProcess(0x10, false, pid);
                    // Read page by page: the top of a stack is near the end of its memory, and a
                    // read that reaches past it gets nothing at all.
                    byte[] buf = new byte[words * 8];
                    int have = 0;
                    while (process != IntPtr.Zero && have < buf.Length) {
                        ulong at = rsp + (ulong)have;
                        int n = Math.Min(buf.Length - have, 4096 - (int)(at & 4095));
                        byte[] part = new byte[n];
                        IntPtr got;
                        if (!ReadProcessMemory(process, (IntPtr)(long)at, part, (IntPtr)n, out got)) break;
                        Array.Copy(part, 0, buf, have, n);
                        have += n;
                    }
                    if (have >= 8) {
                        Stack = new ulong[have / 8];
                        for (int i = 0; i < Stack.Length; i++) Stack[i] = BitConverter.ToUInt64(buf, i * 8);
                    }
                }
            }
            ResumeThread(h);
        }
        Marshal.FreeHGlobal(raw);
        CloseHandle(h);
        return rip;
    }
}
"@

$p = Get-Process $Process -ErrorAction Stop | Select-Object -First 1
$base = [uint64]$p.MainModule.BaseAddress.ToInt64()
$size = [uint64]$p.MainModule.ModuleMemorySize   # the program's image, however big
$preferred = [uint64]0x140000000   # the image base the map is written for

# The map's symbols, by address.
$addrs = New-Object System.Collections.Generic.List[uint64]
$names = New-Object System.Collections.Generic.List[string]
$rows = foreach ($line in [System.IO.File]::ReadLines($Map)) {
    if ($line -match '^\s*[0-9a-f]{4}:[0-9a-f]{8}\s+(\S+)\s+([0-9a-f]{16})\s') {
        [pscustomobject]@{ Va = [Convert]::ToUInt64($Matches[2], 16); Name = $Matches[1] }
    }
}
foreach ($r in ($rows | Sort-Object Va)) { $addrs.Add($r.Va); $names.Add($r.Name) }

$modules = $p.Modules | ForEach-Object {
    [pscustomobject]@{
        Name = $_.ModuleName
        Lo = [uint64]$_.BaseAddress.ToInt64()
        Hi = [uint64]$_.BaseAddress.ToInt64() + [uint64]$_.ModuleMemorySize
    }
}

function Where-Is([uint64]$rip) {
    if ($rip -eq 0) { return "(no context)" }
    if ($rip -lt $base -or $rip -ge $base + $size) {
        foreach ($m in $modules) {
            if ($rip -ge $m.Lo -and $rip -lt $m.Hi) { return "{0}+{1:x}" -f $m.Name, ($rip - $m.Lo) }
        }
        return "(unknown module: {0:x})" -f $rip
    }
    $va = $rip - $base + $preferred
    $i = $addrs.BinarySearch($va)
    if ($i -lt 0) { $i = (-bnot $i) - 1 }
    if ($i -lt 0) { return "(below the image: {0:x})" -f $rip }
    $off = $va - $addrs[$i]
    if ($off -gt 0x100000) { return "(not in a function: {0:x})" -f $rip }
    return "{0}+{1:x}" -f $names[$i], $off
}

$counts = @{}
$cpu0 = @{}
foreach ($t in $p.Threads) { $cpu0[$t.Id] = $t.TotalProcessorTime.TotalMilliseconds }
$wall0 = Get-Date
for ($n = 0; $n -lt $Samples; $n++) {
    $p.Refresh()
    foreach ($t in $p.Threads) {
        if ($OnlyRunning -and $t.ThreadState -ne "Running") { continue }
        $rip = [ThreadIp]::Rip([uint32]$t.Id, [uint32]$p.Id, $(if ($Callers) { 1536 } else { 0 }))
        $where = Where-Is $rip
        if ($Callers -and $rip -ne 0 -and ($rip -lt $base -or $rip -ge $base + $size) -and [ThreadIp]::Stack) {
            $found = 0
            foreach ($w in [ThreadIp]::Stack) {
                if ($w -ge $base -and $w -lt $base + $size) {
                    $where += "  <- " + (Where-Is $w)
                    $found++
                    if ($found -ge $Depth) { break }
                }
            }
        }
        $key = "{0}  {1}" -f $t.Id, $where
        if ($counts.ContainsKey($key)) { $counts[$key]++ } else { $counts[$key] = 1 }
    }
    Start-Sleep -Milliseconds $IntervalMs
}
$counts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First $Top |
    ForEach-Object { "{0,5}  {1}" -f $_.Value, $_.Key }
# The processor time each thread used while the samples were taken: how busy it really was, whether
# the samples caught it running or not.
$wall = ((Get-Date) - $wall0).TotalMilliseconds
$p.Refresh()
$busy = foreach ($t in $p.Threads) {
    if ($cpu0.ContainsKey($t.Id)) {
        $used = $t.TotalProcessorTime.TotalMilliseconds - $cpu0[$t.Id]
        if ($used -ge 1) { [pscustomobject]@{ Id = $t.Id; Used = $used } }
    }
}
"cpu time over {0:f0} ms of sampling:" -f $wall
$busy | Sort-Object Used -Descending | Select-Object -First 8 | ForEach-Object { "  thread {0}: {1:f0} ms ({2:f0} %)" -f $_.Id, $_.Used, (100 * $_.Used / $wall) }
