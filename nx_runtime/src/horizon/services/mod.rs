//! The services this runtime implements.
//!
//! Every service is a Rust object answering IPC requests (see [`super::service`]). What a command
//! does is written from the Switchbrew page of the service; where the page leaves a value open, the
//! choice is recorded in docs/ASSUMPTIONS.md. A command nobody implemented stops the run.

pub(super) mod acc;
mod aoc;
mod apm;
mod applet;
mod audren;
mod bsd;
mod ectx;
mod friend;
mod fsp;
mod hid;
mod lm;
mod mm;
mod nfp;
mod nifm;
mod nvdrv;
mod pctl;
mod pl;
mod prepo;
pub(super) mod savefs;
mod set;
mod ssl;
mod time;
mod vi;

use super::kernel::Kernel;
use super::service::Service;
use std::sync::{Arc, Weak};

/// Register `make` as the factory of the service `name`. Every connection gets a fresh object,
/// which holds a weak reference to the kernel of the process it serves.
fn register<S: Service + 'static>(
    kernel: &Arc<Kernel>,
    name: &str,
    make: impl Fn(Weak<Kernel>) -> S + Send + Sync + 'static,
) {
    let weak = Arc::downgrade(kernel);
    kernel.register_service(name, Box::new(move || Box::new(make(weak.clone()))));
}

/// The names of the services `register_defaults` registers, sorted.
pub fn implemented_names() -> Vec<String> {
    use super::kernel::{ProcessInfo, ProcessKind};
    let kernel = Kernel::new(ProcessInfo::new("names", ProcessKind::Nso, None));
    register_defaults(&kernel);
    kernel.implemented_services()
}

/// Register every implemented service with a kernel. `sm:` is not here: it is the named port the
/// kernel itself answers.
pub fn register_defaults(kernel: &Arc<Kernel>) {
    let applet = applet::Shared::default();
    let gpu = nvdrv::Shared::new();
    let display = Arc::new(vi::Shared::new());
    display.set_gpu(gpu.clone());
    applet.lock().unwrap_or_else(|e| e.into_inner()).display = Some(display.clone());
    register(kernel, "vi:u", move |k| {
        vi::RootService::new(k, display.clone())
    });
    register(kernel, "appletOE", move |k| {
        applet::AppletOe::new(k, applet.clone())
    });
    register(kernel, "acc:u0", acc::AccountService::new);
    register(kernel, "aoc:u", aoc::AddOnContent::new);
    register(kernel, "audren:u", audren::Manager::new);
    register(kernel, "bsd:u", |_| bsd::Client::new());
    register(kernel, "ectx:aw", |_| ectx::WriterForApplication);
    register(kernel, "apm", apm::Apm::new);
    let fs = std::sync::Arc::new(fsp::Shared::default());
    register(kernel, "friend:u", |_| friend::ServiceCreator);
    register(kernel, "fsp-srv", move |k| {
        fsp::FileSystemProxy::new(k, fs.clone())
    });
    let input = Arc::new(hid::Shared::default());
    register(kernel, "hid", move |k| {
        hid::HidServer::new(k, input.clone())
    });
    register(kernel, "lm", lm::LogService::new);
    register(kernel, "mm:u", |_| mm::Requests::new());
    register(kernel, "nfp:user", nfp::UserManager::new);
    register(kernel, "nifm:u", nifm::StaticService::new);
    register(kernel, "ngc:u", |_| Bare("nn::ngc::detail::IService"));
    let report = gpu.clone();
    kernel.on_finish(Box::new(move || report.report()));
    register(kernel, "nvdrv", move |k| nvdrv::NvDrv::new(k, gpu.clone()));
    register(kernel, "pctl", pctl::Factory::new);
    register(kernel, "prepo:u", |_| prepo::Prepo::new());
    let fonts = Arc::new(pl::Shared::default());
    register(kernel, "pl:u", move |k| {
        pl::SharedResourceManager::new(k, fonts.clone())
    });
    register(kernel, "set", set::SettingsServer::new);
    register(kernel, "ssl", |_| ssl::SslService::new());
    let clocks = Arc::new(time::Shared::default());
    register(kernel, "time:u", move |k| {
        time::StaticService::new(k, clocks.clone())
    });
}

/// An interface object with no command implemented: handing it out is right, using it stops the
/// run with the command named.
#[derive(Debug)]
struct Bare(&'static str);

impl Service for Bare {
    fn name(&self) -> &str {
        self.0
    }

    fn request(&mut self, req: &super::service::Request) -> super::service::ServiceResult {
        Err(super::service::Stop::unknown_command(self.0, req.command))
    }
}
