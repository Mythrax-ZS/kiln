//! The services this runtime implements.
//!
//! Every service is a Rust object answering IPC requests (see [`super::service`]). What a command
//! does is written from the Switchbrew page of the service; where the page leaves a value open, the
//! choice is recorded in docs/ASSUMPTIONS.md. A command nobody implemented stops the run.

mod aoc;
mod apm;
mod applet;
mod fsp;
mod lm;
mod pctl;

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
    register(kernel, "appletOE", move |k| {
        applet::AppletOe::new(k, applet.clone())
    });
    register(kernel, "aoc:u", aoc::AddOnContent::new);
    register(kernel, "apm", apm::Apm::new);
    let fs = std::sync::Arc::new(fsp::Shared::default());
    register(kernel, "fsp-srv", move |k| {
        fsp::FileSystemProxy::new(k, fs.clone())
    });
    register(kernel, "lm", lm::LogService::new);
    register(kernel, "pctl", pctl::Factory::new);
}
