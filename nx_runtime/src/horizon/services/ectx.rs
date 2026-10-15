//! `ectx:aw`, the error context service for applications (Switchbrew, "Glue services":
//! `nn::err::context::IWriterForApplication`, 11.0.0 and later).
//!
//! An error context is what a program attaches to an error so that a report of the error can carry
//! it. The console here is offline and writes no reports (docs/ASSUMPTIONS.md A-072, A-133):
//! `CreateContextRegistrar` answers a registrar whose `Complete` is accepted, and every other
//! command, `CommitContext` among them (the page gives it no arguments), stops the run.

use super::super::service::{Request, Response, Service, ServiceResult, Stop};

/// `nn::err::context::IWriterForApplication`.
#[derive(Debug, Default)]
pub struct WriterForApplication;

impl Service for WriterForApplication {
    fn name(&self) -> &str {
        "nn::err::context::IWriterForApplication"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateContextRegistrar() -> IContextRegistrar
            0 => Ok(Response::ok().with_object(Box::new(ContextRegistrar))),
            other => Err(Stop::unknown_command("ectx:aw", other)),
        }
    }
}

/// `nn::err::context::IContextRegistrar`.
#[derive(Debug)]
struct ContextRegistrar;

impl Service for ContextRegistrar {
    fn name(&self) -> &str {
        "nn::err::context::IContextRegistrar"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Complete(): there is no report to add the context to.
            0 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("IContextRegistrar", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: u32) -> Request {
        Request {
            command,
            ..Request::default()
        }
    }

    #[test]
    fn a_registrar_is_made_and_completed_and_other_commands_stop_the_run() {
        let mut w = WriterForApplication;
        let r = w.request(&req(0)).unwrap();
        assert_eq!(r.objects.len(), 1);
        let mut registrar = r.objects.into_iter().next().unwrap();
        assert!(registrar.request(&req(0)).unwrap().result.is_success());
        let stop = registrar.request(&req(1)).unwrap_err();
        assert!(stop.0.contains("IContextRegistrar") && stop.0.contains("command 1"));
        let stop = w.request(&req(1)).unwrap_err();
        assert!(
            stop.0.contains("ectx:aw") && stop.0.contains("command 1"),
            "{}",
            stop.0
        );
    }
}
