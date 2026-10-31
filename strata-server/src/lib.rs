//! Strata over the network.
//!
//! * [`service`]: the operations, over a local database or, as a
//!   coordinator ([`coord`]), over shard servers.
//! * [`http`]: the JSON API (`/v1/...`), Prometheus metrics (`/metrics`)
//!   and the command center ([`ui`], at `/`).
//! * [`grpc`]: the same operations over gRPC (`proto/strata.proto`), which
//!   a coordinator uses to reach its shards; [`plan_pb`]: plans as protobuf.
//! * [`agent`]: answers a question in words by letting a Claude model
//!   query the table.
//! * [`replication`]: a primary shipping its writes to its replicas.
//! * [`media`]: fetching the files MEDIA fields name over HTTP(S).
//! * [`keys`]: API keys with scopes, and the audit log; [`alerts`]: alert
//!   rules over the series.
//! * [`obs`]: logs and request ids; [`metrics`]: what `/metrics` serves;
//!   [`telemetry`]: the running and latest queries, their traces, and the
//!   series of rates and gauges the UI draws; [`logstore`]: the log lines
//!   and queries kept in a Strata database of the server's own, and
//!   searched across a cluster.

pub mod agent;
pub mod alerts;
pub mod auth;
pub mod coord;
pub mod demo;
pub mod grpc;
pub mod heap;
pub mod http;
pub mod idem;
pub mod json;
pub mod keys;
pub mod keysync;
pub mod logstore;
pub mod media;
pub mod metrics;
pub mod obs;
pub mod operator;
pub mod plan_pb;
pub mod profiler;
pub mod profiling;
pub mod replication;
pub mod server;
pub mod service;
pub mod slots;
pub mod telemetry;
pub mod ui;

pub mod pb {
    tonic::include_proto!("strata.v1");
}

/// An error with the HTTP status it answers with.
#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    pub fn bad(m: impl Into<String>) -> ApiError {
        ApiError { status: 400, message: m.into() }
    }

    pub fn internal(e: impl std::fmt::Display) -> ApiError {
        ApiError { status: 500, message: e.to_string() }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<strata::Error> for ApiError {
    fn from(e: strata::Error) -> ApiError {
        let status = match &e {
            strata::Error::Query(_) | strata::Error::Schema(_) => 400,
            strata::Error::Busy(_) => 503,
            // As Elasticsearch's circuit breakers.
            strata::Error::Memory(_) => 429,
            strata::Error::Timeout(_) => 504,
            // A hosted embedding model that refused or did not answer.
            strata::Error::Unavailable(_) => 503,
            strata::Error::Io(_) | strata::Error::Corrupt(_) => 500,
        };
        ApiError { status, message: e.to_string() }
    }
}

impl From<tonic::Status> for ApiError {
    fn from(s: tonic::Status) -> ApiError {
        use tonic::Code;
        let status = match s.code() {
            Code::InvalidArgument => 400,
            Code::FailedPrecondition => 409,
            Code::DeadlineExceeded => 504,
            Code::ResourceExhausted => 429,
            Code::Unavailable => 503,
            Code::Unimplemented => 501,
            _ => 502,
        };
        ApiError { status, message: s.message().to_string() }
    }
}

impl From<ApiError> for tonic::Status {
    fn from(e: ApiError) -> tonic::Status {
        use tonic::Code;
        let code = match e.status {
            400 => Code::InvalidArgument,
            401 => Code::Unauthenticated,
            // A credential's scope does not allow what the call does.
            403 => Code::PermissionDenied,
            // Refused in the state the server is in: an older slot map, a
            // write with the same idempotency key still running.
            409 => Code::FailedPrecondition,
            // Too many queries running: another replica would only be
            // loaded more ([`crate::coord`] does not retry it elsewhere).
            429 => Code::ResourceExhausted,
            // Not on this server: a profile without its sampler.
            501 => Code::Unimplemented,
            503 => Code::Unavailable,
            504 => Code::DeadlineExceeded,
            _ => Code::Internal,
        };
        tonic::Status::new(code, e.message)
    }
}
