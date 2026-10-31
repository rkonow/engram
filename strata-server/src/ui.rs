//! The command center: the page the server answers at `/`, and the
//! scripts and stylesheet it loads from `/ui/`, all compiled into the
//! binary (no CDN: it works offline and behind a firewall). The console it
//! grew from is at `/classic`.
//!
//! With `STRATA_UI_DIR` set to a checkout's `strata-server/ui`, the server
//! reads the files from there on every request instead, so that an edit
//! shows on a reload without a rebuild.

use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};

pub const INDEX: &str = include_str!("../ui/index.html");
pub const CLASSIC: &str = include_str!("../ui/classic.html");

const JS: &str = "text/javascript; charset=utf-8";
const CSS: &str = "text/css; charset=utf-8";

/// Each file under `/ui/`: its path there, type and contents.
pub const FILES: &[(&str, &str, &str)] = &[
    ("app.css", CSS, include_str!("../ui/app/app.css")),
    ("core.js", JS, include_str!("../ui/app/core.js")),
    ("charts.js", JS, include_str!("../ui/app/charts.js")),
    ("data.js", JS, include_str!("../ui/app/data.js")),
    ("main.js", JS, include_str!("../ui/app/main.js")),
    ("sqledit.js", JS, include_str!("../ui/app/sqledit.js")),
    ("catalog.js", JS, include_str!("../ui/app/catalog.js")),
    ("results.js", JS, include_str!("../ui/app/results.js")),
    ("profile.js", JS, include_str!("../ui/app/profile.js")),
    ("flame.js", JS, include_str!("../ui/app/flame.js")),
    ("views/index.js", JS, include_str!("../ui/app/views/index.js")),
    ("views/overview.js", JS, include_str!("../ui/app/views/overview.js")),
    ("views/topology.js", JS, include_str!("../ui/app/views/topology.js")),
    ("views/cluster.js", JS, include_str!("../ui/app/views/cluster.js")),
    ("views/performance.js", JS, include_str!("../ui/app/views/performance.js")),
    ("views/alerts.js", JS, include_str!("../ui/app/views/alerts.js")),
    ("views/queries.js", JS, include_str!("../ui/app/views/queries.js")),
    ("views/profile.js", JS, include_str!("../ui/app/views/profile.js")),
    ("views/logs.js", JS, include_str!("../ui/app/views/logs.js")),
    ("views/access.js", JS, include_str!("../ui/app/views/access.js")),
    ("views/system.js", JS, include_str!("../ui/app/views/system.js")),
    ("views/sql.js", JS, include_str!("../ui/app/views/sql.js")),
    ("views/builder.js", JS, include_str!("../ui/app/views/builder.js")),
    ("views/schema.js", JS, include_str!("../ui/app/views/schema.js")),
    ("views/import.js", JS, include_str!("../ui/app/views/import.js")),
    ("views/relevance.js", JS, include_str!("../ui/app/views/relevance.js")),
    ("views/recsys.js", JS, include_str!("../ui/app/views/recsys.js")),
    ("views/sid.js", JS, include_str!("../ui/app/views/sid.js")),
    ("views/autocomplete.js", JS, include_str!("../ui/app/views/autocomplete.js")),
    ("views/media.js", JS, include_str!("../ui/app/views/media.js")),
    ("views/graph.js", JS, include_str!("../ui/app/views/graph.js")),
    ("views/classic.js", JS, include_str!("../ui/app/views/classic.js")),
];

/// A file from `STRATA_UI_DIR`, when it is set: `rel` under it.
fn from_disk(rel: &str) -> Option<String> {
    let dir = std::env::var_os("STRATA_UI_DIR")?;
    std::fs::read_to_string(std::path::Path::new(&dir).join(rel)).ok()
}

/// The page; the browser asks again each time (`no-cache`), so a new
/// binary's files are seen at once.
pub async fn index() -> Response {
    ([(header::CACHE_CONTROL, "no-cache")], Html(from_disk("index.html").unwrap_or_else(|| INDEX.into()))).into_response()
}

pub async fn classic() -> Response {
    ([(header::CACHE_CONTROL, "no-cache")], Html(from_disk("classic.html").unwrap_or_else(|| CLASSIC.into()))).into_response()
}

pub async fn file(Path(path): Path<String>) -> Response {
    match FILES.iter().find(|(p, ..)| *p == path) {
        Some((p, ty, body)) => {
            let body = from_disk(&format!("app/{p}")).unwrap_or_else(|| (*body).into());
            ([(header::CONTENT_TYPE, *ty), (header::CACHE_CONTROL, "no-cache")], body).into_response()
        }
        None => (StatusCode::NOT_FOUND, "no such file in the command center").into_response(),
    }
}
