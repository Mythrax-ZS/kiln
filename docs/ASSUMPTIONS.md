# Assumed and undocumented behavior

Every behavior here is something the implementation relies on that is not stated authoritatively
in a source the project can quote, or that is inferred. Each entry names its source. Entries marked
UNVERIFIED have not been checked against real hardware or a real title.

| ID | Behavior | Source | Status |
|----|----------|--------|--------|
| A-001 | `CNTFRQ_EL0` reads 19,200,000 Hz and `CNTPCT_EL0` ticks at that rate. | Switchbrew, SVC page, GetSystemTick: "The frequency is 19200000 Hz (constant from official sw)". The page does not say what the hardware register itself returns under Horizon, so equality with `CNTFRQ_EL0` is assumed. | UNVERIFIED |
| A-002 | RomFS image layout: a 0x50-byte header of ten little-endian u64 fields (header size, dir hash table offset/size, dir meta table offset/size, file hash table offset/size, file meta table offset/size, file data offset); directory entries are `parent, sibling, childDir, childFile, nextHash, nameLen` (six u32) followed by a UTF-8 name; file entries are `parent, sibling, dataOff (u64), dataSize (u64), nextHash, nameLen` followed by a UTF-8 name; `0xFFFFFFFF` means no entry; entries start on 4-byte boundaries. | Struct definitions in libnx `nx/include/switch/runtime/devices/romfs_dev.h` (switchbrew/libnx, fetched 2026-10-12). The Switchbrew wiki has no reachable RomFS page. | Documented in libnx; tested only against our own builder |
| A-003 | PFS0 layout: `PFS0` magic, entry count, string table size, 4 reserved bytes, then 0x18-byte entries (offset u64 relative to the data area, size u64, string offset u32, 4 reserved), the string table, then file data. | Switchbrew, NCA page, "PFS0" section. | Documented |
