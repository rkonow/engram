//! Strata: an embedded, single-file search database.
//!
//! One file holds a table whose documents are indexed three ways: posting
//! lists (full text and keyword filters), a columnar forward index (structured
//! fields, doc store) and vector ANN indexes. Queries are SQL with search
//! functions, planned by a cost-based optimizer. See `DESIGN.md`.
//!
//! * [`schema`], [`value`]: tables, field types, documents; [`ingest`]:
//!   rows from JSON, CSV and files; [`timestamp`]: TIMESTAMP values as
//!   text; `sqlite` (feature `sqlite`): SQLite databases in and out.
//! * [`embed`]: the models that fill VECTOR fields declared with `embed`;
//!   [`media`]: MEDIA values, and the files they name.
//! * [`segment`]: the immutable unit of storage, a directory of aligned,
//!   named sections, and the builder that turns documents into one.
//! * [`storage`]: the single-file pager, commit protocol, WAL, snapshots.
//! * [`text`], [`forward`], [`vector`]: the three index families, and
//!   [`graph`]: edges between rows, with traversals.
//! * [`postings`]: the compressed posting-list formats the text and keyword
//!   indexes are built on (Loom, Pyramid, BP128 and the formats they were
//!   compared with), their set-operation algorithms, and the testbed of the
//!   `strata-posting-bench` binary.
//! * [`query`]: SQL front end, planner and executor.
//! * [`cluster`]: merging the answers of shards of one collection.
//! * [`data`], [`metrics`]: BEIR loaders and retrieval metrics for the
//!   evaluation.
//! * [`workload`]: the query workloads shared with the baselines, and the
//!   measurement harness; [`baselines`] (feature `baselines`): SQLite and
//!   Tantivy behind the same interface.

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "baselines")]
pub mod baselines;
pub mod beir;
#[cfg(feature = "audio")]
pub mod clap;
pub mod cluster;
pub mod data;
pub mod db;
pub mod derive;
pub mod embed;
pub mod error;
pub mod forward;
pub mod graph;
pub mod hosted;
pub mod ingest;
pub mod iostat;
pub mod media;
pub mod metrics;
pub mod postings;
pub mod query;
pub mod rank_log;
pub mod schema;
pub mod segment;
pub mod sid;
pub mod simd;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod storage;
pub mod text;
pub mod timestamp;
pub mod upgrade;
pub mod value;
pub mod vector;
pub mod workload;

pub use error::{Error, Result};
pub use schema::{Embed, Field, FieldType, Logical, Schema, VectorCodes, VectorIndexKind};
pub use segment::DocId;
pub use value::{Document, Value};
