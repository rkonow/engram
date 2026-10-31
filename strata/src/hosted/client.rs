//! The HTTP client of hosted models ([`super`]): batching, the gate on
//! requests in flight, retries, the answers of each API, and the cache of
//! query vectors.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use serde_json::{json, Value as J};

use super::{Api, Endpoint, QUERY_CACHE};
use crate::embed::{Embedder, Usage};
use crate::error::{Error, Result};

/// The largest answer read, in bytes: a batch of 256 vectors of 3,072
/// numbers in JSON is about 16 MB.
const MAX_ANSWER: u64 = 256 << 20;

/// A counting semaphore: the requests of a model in flight.
struct Gate {
    free: Mutex<usize>,
    freed: Condvar,
}

impl Gate {
    fn new(n: usize) -> Gate {
        Gate { free: Mutex::new(n.max(1)), freed: Condvar::new() }
    }

    fn enter(&self) -> Pass<'_> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        while *free == 0 {
            free = self.freed.wait(free).unwrap_or_else(|e| e.into_inner());
        }
        *free -= 1;
        Pass(self)
    }
}

struct Pass<'a>(&'a Gate);

impl Drop for Pass<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        self.0.freed.notify_one();
    }
}

/// Query texts' vectors, the oldest dropped first.
struct Cache {
    map: HashMap<String, (Vec<f32>, u64)>,
    tick: u64,
}

/// A model behind an HTTP API.
pub struct HostedEmbedder {
    ep: Endpoint,
    /// The dimensions asked for, or made.
    dim: usize,
    /// Whether requests ask for `dim` dimensions.
    ask_dim: bool,
    agent: ureq::Agent,
    gate: Gate,
    cache: Mutex<Cache>,
    requests: AtomicU64,
    retried: AtomicU64,
    failed: AtomicU64,
    texts: AtomicU64,
    tokens: AtomicU64,
}

/// Why a request did not get vectors.
enum Failure {
    /// Worth another try after the pause, if one is given.
    Again(String, Option<Duration>),
    /// A text too long for the model: cut the batch's texts and try again.
    TooLong(String),
    Fatal(String),
}

impl HostedEmbedder {
    /// The model `ep` describes, making `dim` dimensions: the field's, which
    /// it is asked for when fewer than it makes and it can shorten them.
    /// Without `dim` its own, which Strata must know.
    pub fn new(ep: Endpoint, dim: Option<usize>) -> Result<HostedEmbedder> {
        let bad = |m: String| Error::Schema(format!("model {}: {m}", ep.name));
        let (dim, ask_dim) = match (dim, ep.native_dim) {
            (Some(d), Some(n)) if d == n => (d, false),
            (Some(d), Some(n)) if ep.shortens && d < n => (d, true),
            (Some(d), Some(n)) => return Err(bad(format!("makes {n}-dimension vectors{}; the field holds {d}", if ep.shortens { " (or fewer, on request)" } else { "" }))),
            (Some(d), None) => (d, ep.shortens),
            (None, Some(n)) => (n, false),
            (None, None) => return Err(bad("its dimensions are not known: give the field's dim, or \"dim\" in endpoint.json".into())),
        };
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(ep.timeout)).http_status_as_error(false).user_agent(format!("strata/{}", env!("CARGO_PKG_VERSION"))).build().into();
        Ok(HostedEmbedder {
            gate: Gate::new(ep.concurrency),
            ep,
            dim,
            ask_dim,
            agent,
            cache: Mutex::new(Cache { map: HashMap::new(), tick: 0 }),
            requests: AtomicU64::new(0),
            retried: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            texts: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
        })
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.ep
    }

    /// Vectors of `texts`, documents or queries, in batches sent
    /// `concurrency` at a time. Empty texts get zero vectors unasked.
    fn vectors(&self, texts: &[&str], query: bool) -> Result<Vec<Vec<f32>>> {
        let prefix = if query { &self.ep.query_prefix } else { &self.ep.document_prefix };
        let mut out = vec![Vec::new(); texts.len()];
        let todo: Vec<usize> = (0..texts.len()).filter(|&i| !texts[i].trim().is_empty()).collect();
        for i in (0..texts.len()).filter(|&i| texts[i].trim().is_empty()) {
            out[i] = vec![0.0; self.dim];
        }
        let inputs: Vec<String> = todo.iter().map(|&i| format!("{prefix}{}", cut(texts[i], self.ep.max_chars))).collect();
        let batches = batches(&inputs, self.ep.batch, self.ep.batch_chars);
        let next = AtomicUsize::new(0);
        let work = || -> Result<Vec<(usize, Vec<Vec<f32>>)>> {
            let mut done = Vec::new();
            loop {
                let b = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = batches.get(b) else { return Ok(done) };
                done.push((r.start, self.batch(&inputs[r.clone()], query)?));
            }
        };
        let threads = self.ep.concurrency.min(batches.len()).max(1);
        let parts: Vec<Result<Vec<(usize, Vec<Vec<f32>>)>>> = match threads {
            1 => vec![work()],
            n => std::thread::scope(|s| {
                let hs: Vec<_> = (0..n).map(|_| s.spawn(&work)).collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(Error::Unavailable("an embedding request's thread panicked".into())))).collect()
            }),
        };
        for part in parts {
            for (start, vs) in part? {
                for (j, v) in vs.into_iter().enumerate() {
                    out[todo[start + j]] = v;
                }
            }
        }
        self.texts.fetch_add(todo.len() as u64, Ordering::Relaxed);
        Ok(out)
    }

    /// One batch's vectors, tried again on a rate limit or a server's
    /// failure (twice at most for a query, which someone waits for), and
    /// with its texts cut shorter when one was too long.
    fn batch(&self, inputs: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
        let retries = if query { self.ep.retries.min(2) } else { self.ep.retries };
        let mut cut_to: Option<usize> = None;
        let mut shortened = 0;
        let mut attempt = 0u32;
        loop {
            let owned: Vec<String>;
            let texts: &[String] = match cut_to {
                Some(n) => {
                    owned = inputs.iter().map(|t| cut(t, n).to_string()).collect();
                    &owned
                }
                None => inputs,
            };
            let body = self.request(texts, query);
            let res = {
                let _pass = self.gate.enter();
                self.requests.fetch_add(1, Ordering::Relaxed);
                self.post(&body)
            };
            let failure = match res.and_then(|j| self.parse(&j, texts.len())) {
                Ok(vs) => return Ok(vs),
                Err(f) => f,
            };
            let longest = texts.iter().map(|t| t.chars().count()).max().unwrap_or(0);
            match failure {
                Failure::TooLong(m) if shortened < 4 && longest > 256 => {
                    shortened += 1;
                    cut_to = Some(longest / 2);
                    tracing::warn!(model = %self.ep.name, chars = longest / 2, "a text was too long for the model: cut and sent again ({m})");
                }
                Failure::Again(m, wait) if attempt < retries => {
                    attempt += 1;
                    self.retried.fetch_add(1, Ordering::Relaxed);
                    let pause = wait.unwrap_or_else(|| backoff(attempt)).min(Duration::from_secs(60));
                    tracing::warn!(model = %self.ep.name, attempt, pause_ms = pause.as_millis() as u64, "embedding request failed, trying again: {m}");
                    std::thread::sleep(pause);
                }
                Failure::Again(m, _) => {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                    return Err(Error::Unavailable(format!("model {}: {m} (after {} tries)", self.ep.name, attempt + 1)));
                }
                Failure::TooLong(m) | Failure::Fatal(m) => {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                    return Err(Error::Unavailable(format!("model {}: {m}", self.ep.name)));
                }
            }
        }
    }

    /// The JSON body asking for the vectors of `texts`.
    fn request(&self, texts: &[String], query: bool) -> J {
        let ep = &self.ep;
        let mut body = match ep.api {
            // Floats are every API's default encoding, so none is named.
            Api::OpenAi => json!({"model": ep.model, "input": texts}),
            Api::Cohere => json!({"model": ep.model, "texts": texts, "input_type": if query { "search_query" } else { "search_document" }, "embedding_types": ["float"], "truncate": "END"}),
            Api::Voyage => json!({"model": ep.model, "input": texts, "input_type": if query { "query" } else { "document" }, "truncation": true}),
            Api::Jina => {
                let mut b = json!({"model": ep.model, "input": texts, "truncate": true});
                if ep.model.starts_with("jina-embeddings-v3") || ep.model.starts_with("jina-embeddings-v4") {
                    b["task"] = json!(if query { "retrieval.query" } else { "retrieval.passage" });
                }
                b
            }
            Api::Ollama => json!({"model": ep.model, "input": texts, "truncate": true}),
            Api::Gemini => {
                let model = format!("models/{}", ep.model);
                let task = if query { "RETRIEVAL_QUERY" } else { "RETRIEVAL_DOCUMENT" };
                let reqs: Vec<J> = texts
                    .iter()
                    .map(|t| {
                        let mut r = json!({"model": model, "content": {"parts": [{"text": t}]}, "taskType": task});
                        if self.ask_dim {
                            r[&ep.dim_param] = json!(self.dim);
                        }
                        r
                    })
                    .collect();
                return json!({ "requests": reqs });
            }
        };
        if self.ask_dim {
            body[&ep.dim_param] = json!(self.dim);
        }
        body
    }

    /// Sends `body`, and reads the answer's JSON.
    fn post(&self, body: &J) -> std::result::Result<J, Failure> {
        let ep = &self.ep;
        let mut req = self.agent.post(&ep.url).header("content-type", "application/json");
        if let Some(k) = &ep.key {
            req = match ep.key_header.as_str() {
                "authorization" => req.header("authorization", format!("Bearer {k}")),
                h => req.header(h, k.as_str()),
            };
        }
        for (k, v) in &ep.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let mut res = match req.send(serde_json::to_vec(body).expect("JSON").as_slice()) {
            Ok(r) => r,
            Err(e) => {
                return Err(match e {
                    ureq::Error::BadUri(_) | ureq::Error::InvalidProxyUrl | ureq::Error::Http(_) | ureq::Error::RedirectFailed => Failure::Fatal(format!("{}: {e}", ep.url)),
                    e => Failure::Again(format!("{}: {e}", ep.url), None),
                })
            }
        };
        let status = res.status().as_u16();
        let wait = res.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<f64>().ok()).filter(|s| s.is_finite() && *s >= 0.0).map(Duration::from_secs_f64);
        let bytes = match res.body_mut().with_config().limit(MAX_ANSWER).read_to_vec() {
            Ok(b) => b,
            Err(e) => return Err(Failure::Again(format!("{}: reading the answer: {e}", ep.url), None)),
        };
        let j: Option<J> = serde_json::from_slice(&bytes).ok();
        if (200..300).contains(&status) {
            return j.ok_or_else(|| Failure::Fatal(format!("{} answered {status} with no JSON", ep.url)));
        }
        let said = j.as_ref().and_then(error_message).unwrap_or_else(|| String::from_utf8_lossy(&bytes[..bytes.len().min(300)]).trim().to_string());
        let m = format!("{} answered {status}: {said}", ep.url);
        Err(match status {
            401 | 403 => Failure::Fatal(format!("{m} (the API key in {} was refused)", ep.key_from)),
            408 | 409 | 425 | 429 | 500..=599 => Failure::Again(m, wait),
            400 | 413 | 422 if too_long(&said) => Failure::TooLong(m),
            _ => Failure::Fatal(m),
        })
    }

    /// The `n` vectors an answer holds, in the order of the texts.
    fn parse(&self, j: &J, n: usize) -> std::result::Result<Vec<Vec<f32>>, Failure> {
        let bad = |what: &str| Failure::Fatal(format!("{} answered without {what}", self.ep.url));
        let vectors: Vec<&J> = match self.ep.api {
            Api::OpenAi | Api::Voyage | Api::Jina => {
                let mut data: Vec<&J> = j.get("data").and_then(J::as_array).ok_or_else(|| bad("\"data\""))?.iter().collect();
                if data.iter().all(|d| d.get("index").and_then(J::as_u64).is_some()) {
                    data.sort_by_key(|d| d["index"].as_u64());
                }
                data.into_iter().map(|d| d.get("embedding").ok_or_else(|| bad("\"embedding\""))).collect::<std::result::Result<_, _>>()?
            }
            Api::Cohere => {
                let e = j.get("embeddings").ok_or_else(|| bad("\"embeddings\""))?;
                e.get("float").unwrap_or(e).as_array().ok_or_else(|| bad("\"embeddings\".\"float\""))?.iter().collect()
            }
            Api::Gemini => j.get("embeddings").and_then(J::as_array).ok_or_else(|| bad("\"embeddings\""))?.iter().map(|e| e.get("values").ok_or_else(|| bad("\"values\""))).collect::<std::result::Result<_, _>>()?,
            Api::Ollama => j.get("embeddings").and_then(J::as_array).ok_or_else(|| bad("\"embeddings\""))?.iter().collect(),
        };
        if vectors.len() != n {
            return Err(Failure::Fatal(format!("{} answered {} vectors for {n} texts", self.ep.url, vectors.len())));
        }
        let tokens = ["usage.total_tokens", "usage.prompt_tokens", "meta.billed_units.input_tokens", "prompt_eval_count"].iter().find_map(|p| p.split('.').try_fold(j, |j, k| j.get(k)).and_then(J::as_u64));
        self.tokens.fetch_add(tokens.unwrap_or(0), Ordering::Relaxed);
        vectors
            .into_iter()
            .map(|v| {
                let mut v: Vec<f32> = v.as_array().ok_or_else(|| bad("vectors of numbers"))?.iter().map(|x| x.as_f64().map(|x| x as f32)).collect::<Option<_>>().ok_or_else(|| bad("vectors of numbers"))?;
                if v.len() != self.dim {
                    let hint = if self.ep.shortens || self.ask_dim { "" } else { " (if it makes fewer on request, set \"dimensions\": true in endpoint.json)" };
                    return Err(Failure::Fatal(format!("{} answered {}-dimension vectors; the field holds {}{hint}", self.ep.url, v.len(), self.dim)));
                }
                crate::embed::normalize(&mut v);
                Ok(v)
            })
            .collect()
    }
}

impl Embedder for HostedEmbedder {
    fn name(&self) -> String {
        self.ep.name.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.vectors(texts, false)
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        {
            let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            c.tick += 1;
            let tick = c.tick;
            if let Some((v, t)) = c.map.get_mut(text) {
                *t = tick;
                return Ok(v.clone());
            }
        }
        let v = self.vectors(&[text], true)?.pop().unwrap_or_default();
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.map.len() >= QUERY_CACHE {
            // Drop the oldest quarter.
            let mut ticks: Vec<u64> = c.map.values().map(|(_, t)| *t).collect();
            ticks.sort_unstable();
            let cutoff = ticks[ticks.len() / 4];
            c.map.retain(|_, (_, t)| *t > cutoff);
        }
        let tick = c.tick;
        c.map.insert(text.to_string(), (v.clone(), tick));
        Ok(v)
    }
    fn parallel(&self) -> bool {
        false
    }
    fn usage(&self) -> Option<Usage> {
        Some(Usage {
            requests: self.requests.load(Ordering::Relaxed),
            retries: self.retried.load(Ordering::Relaxed),
            failures: self.failed.load(Ordering::Relaxed),
            texts: self.texts.load(Ordering::Relaxed),
            tokens: self.tokens.load(Ordering::Relaxed),
        })
    }
}

/// The ranges of `inputs` each request sends: at most `n` texts and
/// `chars` characters, and at least one text.
fn batches(inputs: &[String], n: usize, chars: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut size) = (0, 0usize);
    for (i, t) in inputs.iter().enumerate() {
        let len = t.chars().count();
        if i > start && (i - start >= n.max(1) || size.saturating_add(len) > chars) {
            out.push(start..i);
            (start, size) = (i, 0);
        }
        size = size.saturating_add(len);
    }
    if start < inputs.len() {
        out.push(start..inputs.len());
    }
    out
}

/// `t` cut to at most `n` characters, at a character's boundary.
fn cut(t: &str, n: usize) -> &str {
    match t.char_indices().nth(n) {
        Some((at, _)) => &t[..at],
        None => t,
    }
}

/// The pause before try `attempt` (from 1): half a second, doubling, with
/// up to a quarter more at random so that writers do not retry together.
fn backoff(attempt: u32) -> Duration {
    let base = 0.5 * 2f64.powi(attempt.saturating_sub(1).min(6) as i32);
    Duration::from_secs_f64(base * (1.0 + rand::random::<f64>() / 4.0))
}

/// Whether an API's refusal says a text was longer than the model takes.
fn too_long(m: &str) -> bool {
    let m = m.to_ascii_lowercase();
    ["maximum context length", "too long", "too many tokens", "token limit", "exceeds the maximum", "input length", "context length"].iter().any(|p| m.contains(p))
}

/// The message of an API's error answer: OpenAI's and Google's
/// `{"error": {"message"}}`, Ollama's `{"error"}`, Cohere's and Mistral's
/// `{"message"}`, Voyage's and Jina's `{"detail"}`.
fn error_message(j: &J) -> Option<String> {
    let s = j
        .pointer("/error/message")
        .or_else(|| j.get("error").filter(|e| e.is_string()))
        .or_else(|| j.get("message"))
        .or_else(|| j.get("detail"))
        .map(|v| match v {
            J::String(s) => s.clone(),
            v => v.to_string(),
        })?;
    Some(s.chars().take(500).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn dimensions() {
        let ep = |name: &str| Endpoint::from_config(name, None, &env(&[("OPENAI_API_KEY", "k"), ("COHERE_API_KEY", "k")])).unwrap();
        let h = HostedEmbedder::new(ep("openai/text-embedding-3-large"), Some(256)).unwrap();
        assert_eq!((h.dim, h.ask_dim), (256, true));
        assert_eq!(h.request(&["a".into()], false)["dimensions"], 256);
        let h = HostedEmbedder::new(ep("openai/text-embedding-3-large"), None).unwrap();
        assert!(h.request(&["a".into()], false).get("dimensions").is_none());
        let e = HostedEmbedder::new(ep("cohere/embed-english-v3.0"), Some(256)).err().unwrap().to_string();
        assert!(e.contains("makes 1024-dimension vectors; the field holds 256"), "{e}");
        assert!(HostedEmbedder::new(ep("openai/my-own"), None).is_err());
    }

    #[test]
    fn requests_by_api() {
        let ep = |name: &str| Endpoint::from_config(name, None, &env(&[("COHERE_API_KEY", "k"), ("VOYAGE_API_KEY", "k"), ("GEMINI_API_KEY", "k"), ("JINA_API_KEY", "k")])).unwrap();
        let t = ["hi".to_string()];
        let h = HostedEmbedder::new(ep("cohere/embed-v4.0"), Some(512)).unwrap();
        assert_eq!(h.request(&t, true), json!({"model": "embed-v4.0", "texts": ["hi"], "input_type": "search_query", "embedding_types": ["float"], "truncate": "END", "output_dimension": 512}));
        let h = HostedEmbedder::new(ep("voyage/voyage-3.5"), None).unwrap();
        assert_eq!(h.request(&t, false), json!({"model": "voyage-3.5", "input": ["hi"], "input_type": "document", "truncation": true}));
        let h = HostedEmbedder::new(ep("gemini/gemini-embedding-001"), Some(768)).unwrap();
        assert_eq!(h.request(&t, true), json!({"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "hi"}]}, "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768}]}));
        let h = HostedEmbedder::new(ep("jina/jina-embeddings-v3"), None).unwrap();
        assert_eq!(h.request(&t, false)["task"], "retrieval.passage");
    }

    #[test]
    fn batching_and_cutting() {
        let s = |n: usize| "x".repeat(n);
        let inputs = vec![s(5), s(5), s(5), s(20), s(1)];
        assert_eq!(batches(&inputs, 2, 100), [0..2, 2..4, 4..5]);
        assert_eq!(batches(&inputs, 10, 12), [0..2, 2..3, 3..4, 4..5]);
        assert_eq!(batches(&[], 10, 12), Vec::<std::ops::Range<usize>>::new());
        assert_eq!(cut("héllo", 2), "hé");
        assert_eq!(cut("hé", 5), "hé");
        assert!(too_long("This model's maximum context length is 8192 tokens"));
        assert_eq!(error_message(&json!({"error": {"message": "bad key"}})).as_deref(), Some("bad key"));
        assert_eq!(error_message(&json!({"detail": "nope"})).as_deref(), Some("nope"));
    }
}
