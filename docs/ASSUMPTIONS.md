# Assumed and undocumented behavior

Every behavior here is something the implementation relies on that is not stated authoritatively
in a source the project can quote, or that is inferred. Each entry names its source. Entries marked
UNVERIFIED have not been checked against real hardware or a real title.

| ID | Behavior | Source | Status |
|----|----------|--------|--------|
| A-001 | `CNTFRQ_EL0` reads 19,200,000 Hz and `CNTPCT_EL0` ticks at that rate. | Switchbrew, SVC page, GetSystemTick: "The frequency is 19200000 Hz (constant from official sw)". The page does not say what the hardware register itself returns under Horizon, so equality with `CNTFRQ_EL0` is assumed. | UNVERIFIED |
| A-002 | NCA header XTS uses a big-endian sector number as the tweak, sector size 0x200, sectors 0 and 1 for the header and sectors 2 to 5 for the four FS headers (NCA3). | Switchbrew, NCA page, Encryption section ("non-standard tweak, endianness is reversed"). | Documented; tested only against our own encryptor. |
| A-003 | AES-CTR section counter: bytes 0..4 are the FS header SecureValue and bytes 4..8 the Generation, both big-endian; bytes 8..16 are the big-endian absolute NCA byte offset divided by 16. | Switchbrew NCA page lists Generation (0x140) and SecureValue (0x144) but not the counter layout; layout follows the behavior of hactool and Ryujinx/yuzu as remembered, not quoted. | UNVERIFIED against a real NCA |
| A-004 | Key area key derivation when `key_area_key_*_NN` is absent from the keys file: `kek = AES-ECB-decrypt(master_key_NN, aes_kek_generation_source)`, `access = AES-ECB-decrypt(kek, key_area_key_*_source)`, `key = AES-ECB-decrypt(access, aes_key_generation_source)`. | Switchbrew SPL page describes the GenerateAesKek / GenerateAesKey primitives but not the key names; the name mapping follows hactool as remembered. | UNVERIFIED |
| A-005 | Master key index for an NCA is `max(KeyGenerationOld, KeyGeneration)` minus one, with generations 0 and 1 both meaning master key 0. | Switchbrew, NCA page: "KeyGeneration is always MasterKeyVersion + 1, except for generations 0 and 1 which are both version 0." | Documented |
