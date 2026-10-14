# Reads the guest memory of a produced Windows program that is running, from outside: walks the
# program's page table (the `nx_mem_l1` symbol of its link map, see `KILN_LINK_MAP`) with
# ReadProcessMemory. Nothing in the program has to be prepared, and the program is not stopped, so
# a value that is changing may be read torn.
#
# usage: pwsh -File tools/guest_memory.ps1 -Map <map file> -Ranges "<address>:<length>,..." [-Words]
#            [-Find <value>] [-Process title]
#
# Addresses and lengths are hexadecimal with or without 0x. `-Words` prints 64-bit words (one per
# line, with the address), otherwise a hex dump with the characters beside it. `-Find <value>`
# prints instead the addresses in the ranges that hold that 64-bit value (aligned to 8 bytes): a
# scan of gigabytes takes seconds, which finds an object by the address of its table of virtual
# functions.
param(
    [Parameter(Mandatory = $true)][string]$Map,
    [Parameter(Mandatory = $true)][string]$Ranges,
    [string]$Process = "title",
    [switch]$Words,
    [string]$Find = ""
)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Remote {
    [DllImport("kernel32.dll")] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll")] static extern bool ReadProcessMemory(IntPtr h, IntPtr addr, byte[] buf, IntPtr size, out IntPtr read);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
    static IntPtr handle = IntPtr.Zero;
    public static void Open(uint pid) { handle = OpenProcess(0x10 | 0x400, false, pid); if (handle == IntPtr.Zero) throw new Exception("cannot open the process"); }
    public static void Close() { if (handle != IntPtr.Zero) CloseHandle(handle); }
    // Bytes of the host process at an address, or null when they cannot be read.
    public static byte[] Read(ulong addr, int len) {
        byte[] buf = new byte[len];
        IntPtr got;
        if (!ReadProcessMemory(handle, (IntPtr)(long)addr, buf, (IntPtr)len, out got) || (int)got != len) return null;
        return buf;
    }
    public static ulong U64(ulong addr) { byte[] b = Read(addr, 8); return b == null ? 0 : BitConverter.ToUInt64(b, 0); }
    // The guest addresses in [start, start + length) that hold `value` as an aligned 64-bit word,
    // found by walking the page table at `l1` page by page.
    public static System.Collections.Generic.List<ulong> Find(ulong l1, ulong start, ulong length, ulong value) {
        var found = new System.Collections.Generic.List<ulong>();
        ulong l2 = 0, l2Index = ulong.MaxValue;
        byte[] page = new byte[4096];
        for (ulong g = start & ~4095UL; g < start + length; g += 4096) {
            if ((g >> 39) != 0) break;
            ulong index = g >> 21;
            if (index != l2Index) { l2 = U64(l1 + 8 * index); l2Index = index; }
            if (l2 == 0) { g = ((g >> 21) + 1 << 21) - 4096; continue; }
            ulong entry = U64(l2 + 8 * ((g >> 12) & 511));
            if ((entry & 8) == 0) continue;
            IntPtr got;
            if (!ReadProcessMemory(handle, (IntPtr)(long)(entry & 0xFFFFFFFFFFFFF000UL), page, (IntPtr)4096, out got) || (int)got != 4096) continue;
            for (int o = 0; o < 4096; o += 8) {
                if (BitConverter.ToUInt64(page, o) == value) found.Add(g + (ulong)o);
            }
        }
        return found;
    }
}
"@

$p = Get-Process $Process -ErrorAction Stop | Select-Object -First 1
$base = [uint64]$p.MainModule.BaseAddress.ToInt64()
$preferred = [uint64]0x140000000
$l1Symbol = $null
foreach ($line in [System.IO.File]::ReadLines($Map)) {
    if ($line -match '^\s*[0-9a-f]{4}:[0-9a-f]{8}\s+nx_mem_l1\s+([0-9a-f]{16})\s') {
        $l1Symbol = [Convert]::ToUInt64($Matches[1], 16)
        break
    }
}
if ($null -eq $l1Symbol) { throw "nx_mem_l1 is not in the map (is it the map of this build?)" }
[Remote]::Open([uint32]$p.Id)
try {
    $l1 = [Remote]::U64($base + ($l1Symbol - $preferred))
    if ($l1 -eq 0) { throw "the program has not made its guest memory yet" }

    # The host address of the guest byte at $g, or 0 when it is not mapped.
    function Host-Of([uint64]$g) {
        if (($g -shr 39) -ne 0) { return [uint64]0 }
        $l2 = [Remote]::U64($l1 + 8 * ($g -shr 21))
        if ($l2 -eq 0) { return [uint64]0 }
        $entry = [Remote]::U64($l2 + 8 * (($g -shr 12) -band 511))
        if (($entry -band 8) -eq 0) { return [uint64]0 }
        return ($entry -band 0xFFFFFFFFFFFFF000) + ($g -band 0xFFF)
    }
    # Guest bytes, page by page; unmapped parts come back as -1.
    function Guest-Bytes([uint64]$g, [int]$len) {
        $out = New-Object 'int[]' $len
        $i = 0
        while ($i -lt $len) {
            $at = $g + $i
            $chunk = [Math]::Min($len - $i, 4096 - [int]($at -band 0xFFF))
            $host_ = Host-Of $at
            $bytes = $null
            if ($host_ -ne 0) { $bytes = [Remote]::Read($host_, $chunk) }
            for ($k = 0; $k -lt $chunk; $k++) { $out[$i + $k] = if ($null -ne $bytes) { [int]$bytes[$k] } else { -1 } }
            $i += $chunk
        }
        return $out
    }

    foreach ($range in $Ranges.Split(',')) {
        $a, $n = $range.Split(':')
        $addr = [Convert]::ToUInt64($a.Trim().Replace('0x', ''), 16)
        if ($Find -ne "") {
            $length = [Convert]::ToUInt64($n.Trim().Replace('0x', ''), 16)
            $value = [Convert]::ToUInt64($Find.Trim().Replace('0x', ''), 16)
            foreach ($hit in [Remote]::Find($l1, $addr, $length, $value)) { "{0:x}" -f $hit }
            continue
        }
        $len = [int][Convert]::ToUInt64($n.Trim().Replace('0x', ''), 16)
        $bytes = Guest-Bytes $addr $len
        if ($Words) {
            for ($o = 0; $o + 8 -le $len; $o += 8) {
                $v = [uint64]0
                $ok = $true
                for ($k = 7; $k -ge 0; $k--) {
                    if ($bytes[$o + $k] -lt 0) { $ok = $false; break }
                    $v = ($v -shl 8) -bor [uint64]$bytes[$o + $k]
                }
                if ($ok) { "{0:x}: {1:x16}" -f ($addr + $o), $v } else { "{0:x}: (unmapped)" -f ($addr + $o) }
            }
        } else {
            for ($o = 0; $o -lt $len; $o += 16) {
                $hex = ""; $txt = ""
                for ($k = 0; $k -lt 16 -and $o + $k -lt $len; $k++) {
                    $b = $bytes[$o + $k]
                    if ($b -lt 0) { $hex += "-- "; $txt += "?" }
                    else {
                        $hex += "{0:x2} " -f $b
                        $txt += if ($b -ge 32 -and $b -lt 127) { [char]$b } else { "." }
                    }
                }
                "{0:x}: {1,-48} {2}" -f ($addr + $o), $hex, $txt
            }
        }
    }
} finally {
    [Remote]::Close()
}
