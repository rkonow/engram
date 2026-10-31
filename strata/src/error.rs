use std::fmt;

/// Errors of every Strata layer.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// On-disk data failed a check (bad magic, checksum, bounds).
    Corrupt(String),
    /// A schema or document does not fit the table.
    Schema(String),
    /// A query failed to parse, bind or plan.
    Query(String),
    /// Another writer holds the database, or a transaction was misused.
    Busy(String),
    /// A query ran past its deadline.
    Timeout(String),
    /// A query needed more memory than its budget had left
    /// ([`crate::query::memory`]).
    Memory(String),
    /// A service Strata calls refused or did not answer: a hosted
    /// embedding model ([`crate::hosted`]), after its retries.
    Unavailable(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Corrupt(m) => write!(f, "corrupt database: {m}"),
            Error::Schema(m) => write!(f, "schema error: {m}"),
            Error::Query(m) => write!(f, "query error: {m}"),
            Error::Busy(m) => write!(f, "busy: {m}"),
            Error::Timeout(m) => write!(f, "timeout: {m}"),
            Error::Memory(m) => write!(f, "memory: {m}"),
            Error::Unavailable(m) => write!(f, "unavailable: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(any(feature = "sqlite", feature = "baselines"))]
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Query(format!("sqlite: {e}"))
    }
}

/// Shorthand for `Error::Corrupt` from anything printable.
pub fn corrupt(m: impl fmt::Display) -> Error {
    Error::Corrupt(m.to_string())
}
