//! `set`, the settings service a program asks about the console's language and region
//! (Switchbrew, "Settings services").
//!
//! The console is an American English one (docs/ASSUMPTIONS.md A-044); the region is the USA.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `Language` values with their `LanguageCode` strings, in the order Switchbrew lists them; the
/// first fifteen are the original ones, then `zh-Hans` and `zh-Hant` (4.0.0), `pt-BR` (10.1.0).
const LANGUAGES: [&str; 18] = [
    "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW", "en-GB",
    "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
];

/// How many of them the original commands (1 and 3) know.
const ORIGINAL_LANGUAGES: usize = 15;

/// `RegionCode::Usa`.
const REGION_USA: u32 = 1;

/// A `LanguageCode`: the string in a `u64`, zero padded.
fn language_code(language: usize) -> u64 {
    let mut b = [0u8; 8];
    let s = LANGUAGES[language].as_bytes();
    b[..s.len()].copy_from_slice(s);
    u64::from_le_bytes(b)
}

/// `nn::settings::ISettingsServer`.
#[derive(Debug)]
pub struct SettingsServer {
    kernel: Weak<Kernel>,
}

impl SettingsServer {
    pub fn new(kernel: Weak<Kernel>) -> SettingsServer {
        SettingsServer { kernel }
    }

    /// The first `count` language codes, written into the caller's buffer (`out`); the answer
    /// carries the number of codes. `via_pointer` says the buffer is a receive-list entry.
    fn codes(&self, req: &Request, count: usize, via_pointer: bool) -> ServiceResult {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".to_string()))?;
        let out = if via_pointer {
            req.pointer_output(0)
        } else {
            req.output_buffer(0)
        };
        let Some(out) = out else {
            return Err(Stop(
                "the language codes were asked for without a buffer".into(),
            ));
        };
        let n = count.min(out.size as usize / 8);
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| language_code(i).to_le_bytes())
            .collect();
        if !kernel.write_memory(out.addr, &bytes) {
            return Err(Stop(
                "the buffer for the language codes is not writable".into(),
            ));
        }
        let mut r = Response::ok().with_u32(n as u32);
        if via_pointer {
            r = r.with_pointer_reply(0, bytes.len() as u64);
        }
        Ok(r)
    }
}

impl Service for SettingsServer {
    fn name(&self) -> &str {
        "nn::settings::ISettingsServer"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetLanguageCode() -> LanguageCode
            0 => Ok(Response::ok().with_u64(language_code(1))),
            // GetAvailableLanguageCodes() -> (s32 count, pointer buffer of LanguageCode)
            1 => self.codes(req, ORIGINAL_LANGUAGES, true),
            // MakeLanguageCode(s32 Language) -> LanguageCode [4.0.0+]
            2 => {
                let language = req.u32_at(0) as usize;
                if language >= LANGUAGES.len() {
                    return Err(Stop(format!(
                        "MakeLanguageCode of language {language}, which is not in the list"
                    )));
                }
                Ok(Response::ok().with_u64(language_code(language)))
            }
            // GetAvailableLanguageCodeCount() -> s32
            3 => Ok(Response::ok().with_u32(ORIGINAL_LANGUAGES as u32)),
            // GetRegionCode() -> RegionCode
            4 => Ok(Response::ok().with_u32(REGION_USA)),
            // GetAvailableLanguageCodes2() -> (s32 count, buffer of LanguageCode) [4.0.0+]
            5 => self.codes(req, LANGUAGES.len(), false),
            // GetAvailableLanguageCodeCount2() -> s32 [4.0.0+]
            6 => Ok(Response::ok().with_u32(LANGUAGES.len() as u32)),
            other => Err(Stop::unknown_command("set", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // The SDK refuses a settings session with no pointer buffer: `GetAvailableLanguageCodes`
        // returns its array through one (docs/ASSUMPTIONS.md A-067).
        0x1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_codes_are_the_strings_in_a_u64() {
        assert_eq!(&language_code(1).to_le_bytes(), b"en-US\0\0\0");
        assert_eq!(&language_code(14).to_le_bytes(), b"es-419\0\0");
        assert_eq!(&language_code(15).to_le_bytes(), b"zh-Hans\0");
        assert_eq!(LANGUAGES.len(), 18);
    }
}
