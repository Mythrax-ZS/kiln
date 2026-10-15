//! `acc:u0`, the account service an application uses (Switchbrew, "Account services"; the layouts
//! the page does not give are the ones `libnx` uses).
//!
//! The console has one local user, always open, with no linked online account. That is a choice of
//! this runtime (docs/ASSUMPTIONS.md A-066); the identity is fixed so that save data paths and the
//! like are stable between runs. Nothing here talks to a network.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// The user's id: a 128-bit value as two 64-bit words, neither zero. It spells the program's first
/// name ("recomp-user-0001") and stays as it is: a game may have stored it.
pub const USER_ID: [u64; 2] = [0x7265_636f_6d70_2d75, 0x7365_722d_3030_3031];

/// The nickname of the user.
const NICKNAME: &[u8] = b"Kiln";

/// How many users a list holds: `libnx` reads eight and counts the ones that are not zero.
const LIST_SIZE: usize = 8;

fn user_id_bytes() -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&USER_ID[0].to_le_bytes());
    b[8..].copy_from_slice(&USER_ID[1].to_le_bytes());
    b
}

/// The storage `PopLaunchParameter` hands out for the pre-selected user: 0x88 bytes, the magic
/// `0xC79497CA`, a byte saying a user was selected, the user id at offset 8 (what `libnx` reads).
pub fn preselected_user_selection() -> Vec<u8> {
    let mut b = vec![0u8; 0x88];
    b[..4].copy_from_slice(&0xC794_97CAu32.to_le_bytes());
    b[4] = 1;
    b[8..24].copy_from_slice(&user_id_bytes());
    b
}

/// `ProfileBase`: the user id, the time the account was last edited, the nickname (0x38 bytes).
fn profile_base() -> [u8; 0x38] {
    let mut b = [0u8; 0x38];
    b[..0x10].copy_from_slice(&user_id_bytes());
    b[0x18..0x18 + NICKNAME.len()].copy_from_slice(NICKNAME);
    b
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// The list of users `ListAllUsers` and `ListOpenUsers` return: the user, then zeros.
fn user_list() -> Vec<u8> {
    let mut list = vec![0u8; 16 * LIST_SIZE];
    list[..16].copy_from_slice(&user_id_bytes());
    list
}

/// `nn::account::IAccountServiceForApplication`.
#[derive(Debug)]
pub struct AccountService {
    kernel: Weak<Kernel>,
}

impl AccountService {
    pub fn new(kernel: Weak<Kernel>) -> AccountService {
        AccountService { kernel }
    }

    /// Is the user id in the request's data (at `offset`) the user this runtime has?
    fn is_the_user(req: &Request, offset: usize) -> bool {
        req.u64_at(offset) == USER_ID[0] && req.u64_at(offset + 8) == USER_ID[1]
    }

    /// Return a list of users through the caller's pointer buffer.
    fn list(&self, req: &Request) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let Some(out) = req.pointer_output(0) else {
            return Err(Stop(
                "a list of users was asked for without a buffer".into(),
            ));
        };
        let list = user_list();
        let n = list.len().min(out.size as usize);
        if !kernel.write_memory(out.addr, &list[..n]) {
            return Err(Stop(
                "the buffer for the list of users is not writable".into(),
            ));
        }
        Ok(Response::ok().with_pointer_reply(0, n as u64))
    }
}

impl Service for AccountService {
    fn name(&self) -> &str {
        "nn::account::IAccountServiceForApplication"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetUserCount() -> s32
            0 => Ok(Response::ok().with_u32(1)),
            // GetUserExistence(Uid) -> bool
            1 => Ok(Response::ok().with_bytes(&[u8::from(Self::is_the_user(req, 0))])),
            // ListAllUsers() -> buffer of Uid; ListOpenUsers() the same: the user is always open.
            // ListOpenContextStoredUsers() [6.0.0+] the same too: the users that were open when the
            // console last stopped are the ones that are open (docs/ASSUMPTIONS.md A-144)
            2 | 3 | 131 => self.list(req),
            // ListQualifiedUsers() [6.0.0+]: the users the program can use; the one user qualifies
            // (docs/ASSUMPTIONS.md A-146)
            141 => self.list(req),
            // GetLastOpenedUser() -> Uid
            4 => Ok(Response::ok().with_bytes(&user_id_bytes())),
            // GetProfile(Uid) -> IProfile
            5 => {
                if !Self::is_the_user(req, 0) {
                    return Err(Stop(format!(
                        "the program asked for the profile of user {:016x}{:016x}, which does not \
                         exist",
                        req.u64_at(8),
                        req.u64_at(0)
                    )));
                }
                Ok(Response::ok().with_object(Box::new(Profile {
                    kernel: self.kernel.clone(),
                })))
            }
            // GetBaasAccountManagerForApplication(Uid) -> IManagerForApplication: the object
            // exists; none of its commands is implemented, because the user has no network
            // service account (docs/ASSUMPTIONS.md A-066), so the first one used stops the run.
            101 => {
                if !Self::is_the_user(req, 0) {
                    return Err(Stop(
                        "the program asked for the account manager of a user that does not exist"
                            .into(),
                    ));
                }
                Ok(Response::ok().with_object(Box::new(BaasManager)))
            }
            // IsUserRegistrationRequestPermitted(u64, pid) -> bool: there is nothing to register
            50 => Ok(Response::ok().with_bytes(&[0])),
            // TrySelectUserWithoutInteraction(bool) -> Uid: the only user is the selection
            51 => Ok(Response::ok().with_bytes(&user_id_bytes())),
            // LoadOpenContext(Uid) [5.0.0+]: opens the user from the context the console stored;
            // the user is always open, so there is nothing to load (docs/ASSUMPTIONS.md A-145)
            130 => {
                if !Self::is_the_user(req, 0) {
                    return Err(Stop(
                        "the program asked to load the open context of a user that does not exist"
                            .into(),
                    ));
                }
                Ok(Response::ok())
            }
            // IsUserAccountSwitchLocked() -> bool [6.0.0+]: the program has no control property
            // that locks the switch, and there is one user anyway (docs/ASSUMPTIONS.md A-138)
            150 => Ok(Response::ok().with_bytes(&[0])),
            // InitializeApplicationInfoV0 (before 6.0.0) / InitializeApplicationInfo(u64, pid).
            // InitializeApplicationInfoV2 (160) is taken to be the same: a title's first call of it
            // had the same input, 8 bytes and the process id (docs/ASSUMPTIONS.md A-137).
            100 | 140 | 160 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("acc:u0", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // `ListAllUsers` and `IProfile::Get` return arrays through pointer buffers.
        0x1000
    }
}

/// `nn::account::profile::IProfile`.
#[derive(Debug)]
struct Profile {
    kernel: Weak<Kernel>,
}

impl Service for Profile {
    fn name(&self) -> &str {
        "nn::account::profile::IProfile"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Get() -> (ProfileBase, buffer UserData of 0x80 bytes)
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(out) = req.pointer_output(0) else {
                    return Err(Stop(
                        "the profile's user data was asked for without a buffer".into(),
                    ));
                };
                let data = [0u8; 0x80];
                let n = data.len().min(out.size as usize);
                if !kernel.write_memory(out.addr, &data[..n]) {
                    return Err(Stop("the buffer for the user data is not writable".into()));
                }
                Ok(Response::ok()
                    .with_bytes(&profile_base())
                    .with_pointer_reply(0, n as u64))
            }
            // GetBase() -> ProfileBase
            1 => Ok(Response::ok().with_bytes(&profile_base())),
            other => Err(Stop::unknown_command("IProfile", other)),
        }
    }
}

/// `nn::account::baas::IManagerForApplication`: the user's network service account, which this
/// user does not have. Every command stops the run and names itself.
#[derive(Debug)]
struct BaasManager;

impl Service for BaasManager {
    fn name(&self) -> &str {
        "nn::account::baas::IManagerForApplication"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop::unknown_command(
            "IManagerForApplication (the user has no network service account)",
            req.command,
        ))
    }
}
