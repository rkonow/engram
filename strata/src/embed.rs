//! Embedders: the models that turn a row's text or image, or a query,
//! into the vector of a VECTOR field declared with
//! [`crate::schema::Embed`]. Every vector is unit length, so Strata's dot
//! product is cosine similarity. Every model embeds text
//! ([`Embedder::embed`]); a model of images and text, such as SigLIP,
//! also embeds decoded images ([`Embedder::embed_inputs`]) into the same
//! space, so a text query finds images and an image finds images.
//!
//! * [`StaticEmbedder`] (feature `embed`, on by default): a Model2Vec model
//!   (token lookup and mean), such as `potion-retrieval-32M`. Tens of
//!   microseconds a row on one core.
//! * [`BertEmbedder`] (feature `bert`): a BERT encoder with CLS pooling
//!   through candle, such as `bge-small-en-v1.5` or
//!   `snowflake-arctic-embed-m-v1.5`. Better vectors, milliseconds a row.
//! * [`SiglipEmbedder`] (feature `vision`): SigLIP's image and text
//!   towers through candle, `siglip-base-patch16-224`. A tenth of a second
//!   an image on a few cores.
//! * [`HashEmbedder`]: feature hashing of words (`hash:<dim>`), and of an
//!   image's colours, for tests and demos without model files; it matches
//!   words and colours, not meanings.
//!
//! [`open`] finds a model by the name a schema gives it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result};

pub trait Embedder: Send + Sync {
    fn name(&self) -> String;
    fn dim(&self) -> usize;
    /// Embeddings of documents, or why the model made none (a model behind
    /// an API that refused, or did not answer).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embedding of a search query (some models prefix an instruction).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text])?.pop().unwrap_or_default())
    }
    /// What it embeds: text, and images for a model of both.
    fn modalities(&self) -> &'static [Modality] {
        &[Modality::Text]
    }
    /// Embeddings of inputs of the kinds it takes ([`Embedder::modalities`]),
    /// in the space of its text vectors: text as documents
    /// ([`Embedder::embed`]), images with the model's image tower.
    fn embed_inputs(&self, inputs: &[Input<'_>]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<&str> = inputs.iter().filter_map(|i| if let Input::Text(t) = i { Some(*t) } else { None }).collect();
        match inputs.iter().find(|i| !matches!(i, Input::Text(_))) {
            Some(i) => Err(model_err(format!("model {} does not embed {}", self.name(), modality_name(i.modality())))),
            None => self.embed(&texts),
        }
    }
    /// Whether many texts are best split across the cores, as a model
    /// computed here is; a model behind an API sends its own requests at
    /// once instead.
    fn parallel(&self) -> bool {
        true
    }
    /// What a model behind an API has asked of it since it opened.
    fn usage(&self) -> Option<Usage> {
        None
    }
}

/// The requests a model behind an API sent ([`Embedder::usage`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub requests: u64,
    /// Requests sent again after a rate limit or a failure.
    pub retries: u64,
    /// Requests that failed for good, failing their write or query.
    pub failures: u64,
    /// Texts embedded.
    pub texts: u64,
    /// Tokens the API counted, where its answers say.
    pub tokens: u64,
}

/// A kind of input a model embeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
        }
    }
}

/// An image decoded to 8-bit RGB, row by row
/// ([`crate::media::decode_image`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Sound decoded to one channel of samples, `rate` a second
/// ([`crate::media::decode_audio`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub rate: u32,
    pub samples: Vec<f32>,
}

/// What [`Embedder::embed_inputs`] embeds.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    Text(&'a str),
    Image(&'a Image),
    Audio(&'a Audio),
}

impl Input<'_> {
    pub fn modality(&self) -> Modality {
        match self {
            Input::Text(_) => Modality::Text,
            Input::Image(_) => Modality::Image,
            Input::Audio(_) => Modality::Audio,
        }
    }
}

/// What a media file is to a model: sound if its bytes are of an audio
/// file, else an image.
pub fn file_modality(data: &[u8]) -> Modality {
    match crate::media::describe(data).mime.is_some_and(|m| m.starts_with("audio/")) {
        true => Modality::Audio,
        false => Modality::Image,
    }
}

/// The vector of an image file (JPEG, PNG, WebP, GIF, BMP) or a sound
/// file (WAV, FLAC, MP3, Ogg Vorbis, AAC) by `e`, a model of images or of
/// sound ([`file_modality`]): a query by a file, as `{"embed": {"image":
/// ...}}` or `{"embed": {"audio": ...}}` gives it.
pub fn embed_file(e: &dyn Embedder, data: &[u8]) -> Result<Vec<f32>> {
    let m = file_modality(data);
    if !e.modalities().contains(&m) {
        return Err(model_err(format!("model {} does not embed {}", e.name(), modality_name(m))));
    }
    let v = match m {
        Modality::Audio => e.embed_inputs(&[Input::Audio(&crate::media::decode_audio(data)?)])?,
        _ => e.embed_inputs(&[Input::Image(&crate::media::decode_image(data)?)])?,
    };
    Ok(v.into_iter().next().unwrap_or_default())
}

/// A query's file, the `embed` of a parameter `{"embed": {"image":
/// "<base64>"}}` or `{"embed": {"audio": "<base64>"}}`: its modality and
/// bytes, which have to be of that kind ([`file_is`]).
pub fn query_file(kind: &str, base64: &str) -> Result<(Modality, Vec<u8>)> {
    let want = match kind {
        "image" => Modality::Image,
        "audio" => Modality::Audio,
        k => return Err(Error::Query(format!("\"embed\" is a text, {{\"image\": \"<base64>\"}} or {{\"audio\": \"<base64>\"}}, not {{{k:?}: ...}}"))),
    };
    let data = crate::media::base64_bytes(base64).map_err(|m| Error::Query(format!("{{\"embed\": {{\"{kind}\": ...}}}} is a file in base64: {m}")))?;
    file_is(want, &data)?;
    Ok((want, data))
}

/// Refuses a query's file given as an image that is a sound file, or given
/// as a sound that is not one.
pub fn file_is(want: Modality, data: &[u8]) -> Result<()> {
    match (want, file_modality(data)) {
        (Modality::Audio, Modality::Audio) | (Modality::Image, Modality::Image) => Ok(()),
        (Modality::Audio, _) => Err(Error::Query("the file given as audio is not a sound file (WAV, FLAC, MP3, Ogg Vorbis, AAC)".into())),
        _ => Err(Error::Query("the file given as an image is a sound file".into())),
    }
}

/// What a model takes, as a query's file names it: "images", "sound".
pub fn modality_name(m: Modality) -> &'static str {
    match m {
        Modality::Text => "text",
        Modality::Image => "images",
        Modality::Audio => "sound",
    }
}

pub(crate) fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn model_err(what: impl std::fmt::Display) -> Error {
    Error::Schema(what.to_string())
}

/// Where [`open`] looks for models by default: `$STRATA_MODELS`, else
/// `~/.local/share/strata/models`.
pub fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("STRATA_MODELS") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join(".local/share/strata/models")
}

/// Models [`open`] knows how to fetch: name, Hugging Face repository, and
/// what it is.
pub const KNOWN_MODELS: &[(&str, &str, &str)] = &[
    ("potion-retrieval-32M", "minishlab/potion-retrieval-32M", "Model2Vec, 512 dimensions, fast (the model engram uses)"),
    ("potion-base-8M", "minishlab/potion-base-8M", "Model2Vec, 256 dimensions, smallest"),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "BERT, 384 dimensions (feature bert)"),
    ("snowflake-arctic-embed-m-v1.5", "Snowflake/snowflake-arctic-embed-m-v1.5", "BERT, 768 dimensions (feature bert)"),
    ("siglip-base-patch16-224", "google/siglip-base-patch16-224", "SigLIP, images and text, 768 dimensions (feature vision)"),
    ("clap-htsat-unfused", "laion/clap-htsat-unfused", "CLAP, sound and text, 512 dimensions (feature audio)"),
];

/// The dimensions of a model [`open`] knows by name (`hash:N` and hosted
/// models included, [`crate::hosted::KNOWN`]), without opening it.
pub fn known_dim(model: &str) -> Option<usize> {
    if let Some(n) = model.strip_prefix("hash:") {
        return n.parse().ok();
    }
    if let Some(k) = crate::hosted::known_model(model) {
        return Some(k.dim);
    }
    let dims = [("potion-retrieval-32M", 512), ("potion-base-8M", 256), ("bge-small-en-v1.5", 384), ("snowflake-arctic-embed-m-v1.5", 768), ("siglip-base-patch16-224", 768), ("clap-htsat-unfused", 512)];
    dims.iter().find(|(m, _)| m.eq_ignore_ascii_case(model)).map(|&(_, d)| d)
}

/// A model [`catalog`] lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listed {
    pub name: String,
    pub dim: Option<usize>,
    /// Behind an API ([`crate::hosted`]), rather than computed here.
    pub hosted: bool,
    pub what: String,
    /// Whether a field naming it would get its model here, or why not (the
    /// files not fetched, the key not set, the feature not built).
    pub ready: std::result::Result<(), String>,
}

/// The models a schema can name, and whether each would open here: those
/// Strata fetches ([`KNOWN_MODELS`]), the hosted models it knows
/// ([`crate::hosted::KNOWN`]), and the other directories of `models` that
/// hold a model or an `endpoint.json`. Nothing is sent or loaded.
pub fn catalog(models: &Path) -> Vec<Listed> {
    let mut out: Vec<Listed> = KNOWN_MODELS
        .iter()
        .map(|&(name, _, what)| {
            let dir = models.join(name);
            let fetched = model_files(name).iter().all(|f| dir.join(f).exists());
            let feature = match name {
                n if n.starts_with("siglip") => (!cfg!(feature = "vision")).then_some("vision"),
                n if n.starts_with("clap") => (!cfg!(feature = "audio")).then_some("audio"),
                n if n.starts_with("bge") || n.starts_with("snowflake") => (!cfg!(feature = "bert")).then_some("bert"),
                _ => None,
            };
            let ready = match (feature, fetched) {
                (Some(f), _) => Err(format!("built without the `{f}` feature")),
                (None, false) => Err(format!("not fetched: strata-server fetch-model {name}")),
                (None, true) => Ok(()),
            };
            Listed { name: name.into(), dim: known_dim(name), hosted: false, what: what.into(), ready }
        })
        .collect();
    let ready_hosted = |name: &str, config: Option<&serde_json::Value>| -> std::result::Result<(), String> {
        let ep = crate::hosted::Endpoint::from_config(name, config, &|k| std::env::var(k).ok()).map_err(|e| e.to_string())?;
        match cfg!(feature = "hosted") {
            true => Ok(()),
            false => Err(format!("a hosted model ({}): build with the `hosted` feature", ep.api.as_str())),
        }
    };
    for k in crate::hosted::KNOWN {
        let provider = k.name.split('/').next().unwrap_or_default();
        let config = std::fs::read_to_string(models.join(k.name).join("endpoint.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        let what = format!("{provider}, {} dimensions{}", k.dim, if k.shortens { " (or fewer)" } else { "" });
        out.push(Listed { name: k.name.into(), dim: Some(k.dim), hosted: true, what, ready: ready_hosted(k.name, config.as_ref()) });
    }
    // Directories of models, and of endpoints, not listed above.
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(models).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    for dir in dirs {
        let Some(name) = dir.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
        if out.iter().any(|l| l.name.eq_ignore_ascii_case(&name)) {
            continue;
        }
        let endpoint = dir.join("endpoint.json");
        if endpoint.exists() {
            let config = std::fs::read_to_string(&endpoint).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
            let dim = config.as_ref().and_then(|c| c.get("dim")).and_then(serde_json::Value::as_u64).map(|d| d as usize);
            let api = config.as_ref().and_then(|c| c.get("api")).and_then(serde_json::Value::as_str).unwrap_or("?").to_string();
            let ready = match &config {
                Some(c) => ready_hosted(&name, Some(c)),
                None => Err(format!("{}: not JSON", endpoint.display())),
            };
            out.push(Listed { name, dim, hosted: true, what: format!("{api} endpoint ({})", endpoint.display()), ready });
        } else if dir.join("model.safetensors").exists() {
            let config: serde_json::Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            // Ranking models and cross-encoders live here too; only embedders are listed.
            let kind = config["model_type"].as_str().unwrap_or("model2vec").to_string();
            if config["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str().is_some_and(|s| s.contains("SequenceClassification")))) {
                continue;
            }
            let dim = config["hidden_size"].as_u64().map(|d| d as usize);
            out.push(Listed { name, dim, hosted: false, what: format!("{kind} model in {}", dir.display()), ready: Ok(()) });
        }
    }
    out
}

/// The files the model directory of `name` needs: CLAP's repository
/// ships its weights as a PyTorch file only.
pub fn model_files(name: &str) -> &'static [&'static str] {
    match name.starts_with("clap") {
        true => &["config.json", "tokenizer.json", "pytorch_model.bin"],
        false => &["config.json", "tokenizer.json", "model.safetensors"],
    }
}

/// Opens the model a schema names: `hash:<dim>`, or a directory under
/// `models` (or an absolute path) holding a Model2Vec, BERT, SigLIP or
/// CLAP model ([`model_files`]), or a hosted model ([`crate::hosted`]):
/// `<provider>/<model>` (`openai/text-embedding-3-small`), or a directory
/// holding an `endpoint.json`. [`open_for`] opens one for a field's
/// dimensions.
pub fn open(model: &str, models: &Path) -> Result<Arc<dyn Embedder>> {
    open_for(model, None, models)
}

/// [`open`], for a field of `dim` dimensions: a hosted model that makes
/// more, and can make fewer, is asked for `dim`, and one Strata does not
/// know makes `dim`. A model computed here makes what it makes, which the
/// field must hold.
pub fn open_for(model: &str, dim: Option<usize>, models: &Path) -> Result<Arc<dyn Embedder>> {
    if let Some(d) = model.strip_prefix("hash:") {
        let d = d.parse().map_err(|_| model_err(format!("model {model}: hash:<dimensions>")))?;
        return Ok(Arc::new(HashEmbedder { dim: d }));
    }
    let dir = if Path::new(model).is_absolute() { PathBuf::from(model) } else { models.join(model) };
    let endpoint = dir.join("endpoint.json");
    let has_weights = dir.join("model.safetensors").exists() || dir.join("pytorch_model.bin").exists();
    if endpoint.exists() || (!has_weights && crate::hosted::is_hosted_name(model)) {
        let config = match endpoint.exists() {
            true => {
                let text = std::fs::read_to_string(&endpoint).map_err(|e| model_err(format!("{}: {e}", endpoint.display())))?;
                Some(serde_json::from_str::<serde_json::Value>(&text).map_err(|e| model_err(format!("{}: {e}", endpoint.display())))?)
            }
            false => None,
        };
        let ep = crate::hosted::Endpoint::from_config(model, config.as_ref(), &|k| std::env::var(k).ok())?;
        #[cfg(feature = "hosted")]
        return Ok(Arc::new(crate::hosted::HostedEmbedder::new(ep, dim)?));
        #[cfg(not(feature = "hosted"))]
        return Err(model_err(format!("model {model} is a hosted model ({}): build with the `hosted` feature", ep.api.as_str())));
    }
    let _ = dim;
    if !has_weights {
        return Err(model_err(format!("model {model}: no {} (fetch it first)", dir.join("model.safetensors").display())));
    }
    let config: serde_json::Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    if config["model_type"] == "bert" {
        #[cfg(feature = "bert")]
        return Ok(Arc::new(BertEmbedder::open(&dir)?));
        #[cfg(not(feature = "bert"))]
        return Err(model_err(format!("model {model} is a BERT model: build with the `bert` feature")));
    }
    if config["model_type"] == "siglip" {
        #[cfg(feature = "vision")]
        return Ok(Arc::new(SiglipEmbedder::open(&dir)?));
        #[cfg(not(feature = "vision"))]
        return Err(model_err(format!("model {model} is a SigLIP model: build with the `vision` feature")));
    }
    if config["model_type"] == "clap" {
        #[cfg(feature = "audio")]
        return Ok(Arc::new(crate::clap::ClapEmbedder::open(&dir)?));
        #[cfg(not(feature = "audio"))]
        return Err(model_err(format!("model {model} is a CLAP model: build with the `audio` feature")));
    }
    #[cfg(feature = "embed")]
    return Ok(Arc::new(StaticEmbedder::open(&dir)?));
    #[cfg(not(feature = "embed"))]
    Err(model_err(format!("model {model}: built without the `embed` feature")))
}

// ---------------------------------------------------------------- static

/// Model2Vec: the embedding of a text is the mean of its tokens' rows.
#[cfg(feature = "embed")]
pub struct StaticEmbedder {
    tokenizer: tokenizers::Tokenizer,
    table: Table,
    dim: usize,
    unk: Option<u32>,
    name: String,
    max_tokens: usize,
}

#[cfg(feature = "embed")]
impl StaticEmbedder {
    pub fn open(dir: &Path) -> Result<StaticEmbedder> {
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| model_err(format!("{}: {e}", dir.join("tokenizer.json").display())))?;
        let table = Table::open(&dir.join("model.safetensors"), "embeddings")?;
        let (rows, dim) = (table.rows, table.dim);
        if rows < tokenizer.get_vocab_size(true) {
            return Err(model_err(format!("{}: embedding table has {rows} rows, tokenizer {} tokens", dir.display(), tokenizer.get_vocab_size(true))));
        }
        let unk = tokenizer.token_to_id("[UNK]");
        let name = format!("static:{}", dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        Ok(StaticEmbedder { tokenizer, table, dim, unk, name, max_tokens: 512 })
    }

    fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let mut n = 0usize;
        for &id in ids.iter().filter(|&&id| Some(id) != self.unk).take(self.max_tokens) {
            if self.table.add_row(id as usize, &mut v) {
                n += 1;
            }
        }
        if n > 0 {
            v.iter_mut().for_each(|x| *x /= n as f32);
            normalize(&mut v);
        }
        v
    }
}

#[cfg(feature = "embed")]
impl Embedder for StaticEmbedder {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| match self.tokenizer.encode(*t, false) {
                Ok(e) => self.embed_ids(e.get_ids()),
                Err(_) => vec![0f32; self.dim],
            })
            .collect())
    }
}

/// One F32 tensor of a safetensors file, mapped rather than read: a process
/// starts without loading the table, and only the rows of tokens it meets
/// are paged in.
#[cfg(feature = "embed")]
struct Table {
    map: memmap2::Mmap,
    /// Byte offset of row 0 in the file.
    start: usize,
    rows: usize,
    dim: usize,
}

#[cfg(feature = "embed")]
impl Table {
    fn open(path: &Path, tensor: &str) -> Result<Table> {
        let file = std::fs::File::open(path).map_err(|e| model_err(format!("{}: {e}", path.display())))?;
        // SAFETY: the model file is not modified while mapped (fetching a
        // model writes a new file).
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| model_err(format!("{}: {e}", path.display())))?;
        let bad = |m: String| model_err(format!("{}: {m}", path.display()));
        if map.len() < 8 {
            return Err(bad("not a safetensors file".into()));
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(map.get(8..8 + hlen).ok_or_else(|| bad("truncated header".into()))?).map_err(|e| bad(e.to_string()))?;
        let t = header.get(tensor).ok_or_else(|| bad(format!("no tensor {tensor}")))?;
        if t["dtype"] != "F32" {
            return Err(bad(format!("{tensor}: dtype {} (need F32)", t["dtype"])));
        }
        let shape: Vec<usize> = t["shape"].as_array().into_iter().flatten().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
        let offs: Vec<usize> = t["data_offsets"].as_array().into_iter().flatten().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
        let [rows, dim] = shape[..] else { return Err(bad(format!("{tensor}: shape {shape:?}"))) };
        let (a, b) = (8 + hlen + offs.first().copied().unwrap_or(0), 8 + hlen + offs.get(1).copied().unwrap_or(0));
        if b > map.len() || b - a != rows * dim * 4 {
            return Err(bad(format!("{tensor}: {} bytes for shape {rows}x{dim}", b.saturating_sub(a))));
        }
        Ok(Table { map, start: a, rows, dim })
    }

    /// Adds row `id` to `acc`; false if there is no such row.
    fn add_row(&self, id: usize, acc: &mut [f32]) -> bool {
        if id >= self.rows {
            return false;
        }
        let off = self.start + id * self.dim * 4;
        for (a, c) in acc.iter_mut().zip(self.map[off..off + self.dim * 4].chunks_exact(4)) {
            *a += f32::from_le_bytes(c.try_into().unwrap());
        }
        true
    }
}

// ---------------------------------------------------------------- hash

/// Lowercased words hashed into `dim` buckets (signed), for tests. An
/// image is the words of its colours: each cell of a 4 by 4 grid is a word
/// of its mean colour, a few levels a channel, so equal images have equal
/// vectors and images of like colours are near.
pub struct HashEmbedder {
    pub dim: usize,
}

impl HashEmbedder {
    fn of_words<'a>(&self, words: impl Iterator<Item = &'a str>) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for w in words {
            let h = fnv(&w.to_lowercase());
            v[(h % self.dim as u64) as usize] += if h >> 63 == 0 { 1.0 } else { -1.0 };
        }
        normalize(&mut v);
        v
    }

    /// The words of an image's colours.
    fn colour_words(image: &Image) -> Vec<String> {
        let (w, h) = (image.width.max(1) as usize, image.height.max(1) as usize);
        let mut sums = [[0u64; 4]; 16];
        for y in 0..image.height as usize {
            for x in 0..image.width as usize {
                let (cell, at) = ((y * 4 / h) * 4 + x * 4 / w, (y * w + x) * 3);
                let Some(p) = image.rgb.get(at..at + 3) else { continue };
                for c in 0..3 {
                    sums[cell][c] += p[c] as u64;
                }
                sums[cell][3] += 1;
            }
        }
        sums.iter().enumerate().filter(|(_, s)| s[3] > 0).map(|(i, s)| format!("c{i}r{}g{}b{}", s[0] / s[3] / 64, s[1] / s[3] / 64, s[2] / s[3] / 64)).collect()
    }
}

impl Embedder for HashEmbedder {
    fn name(&self) -> String {
        format!("hash:{}", self.dim)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.of_words(t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()))).collect())
    }
    fn modalities(&self) -> &'static [Modality] {
        &[Modality::Text, Modality::Image]
    }
    fn embed_inputs(&self, inputs: &[Input<'_>]) -> Result<Vec<Vec<f32>>> {
        inputs
            .iter()
            .map(|i| match i {
                Input::Text(t) => Ok(self.of_words(t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()))),
                Input::Image(image) => Ok(self.of_words(Self::colour_words(image).iter().map(String::as_str))),
                Input::Audio(_) => Err(model_err(format!("model {} does not embed sound", self.name()))),
            })
            .collect()
    }
}

/// 64-bit FNV-1a.
pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------- bert

#[cfg(feature = "bert")]
pub struct BertEmbedder {
    model: candle_transformers::models::bert::BertModel,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    name: String,
}

#[cfg(feature = "bert")]
impl BertEmbedder {
    /// The query instruction of BGE v1.5 and Arctic Embed v1.5.
    const QUERY_PREFIX: &'static str = "Represent this sentence for searching relevant passages: ";

    pub fn open(dir: &Path) -> Result<BertEmbedder> {
        use candle_nn::VarBuilder;
        use candle_transformers::models::bert::{BertModel, Config};
        let e = |x: String| model_err(format!("{}: {x}", dir.display()));
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|x| e(x.to_string()))?;
        let cfg: Config = serde_json::from_str(&text).map_err(|x| e(x.to_string()))?;
        let dim = serde_json::from_str::<serde_json::Value>(&text).ok().and_then(|v| v["hidden_size"].as_u64()).unwrap_or(384) as usize;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|x| e(x.to_string()))?;
        tokenizer.with_truncation(Some(tokenizers::TruncationParams { max_length: 512, ..Default::default() })).map_err(|x| e(x.to_string()))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));
        let dev = candle_core::Device::Cpu;
        // SAFETY: the weights file is not modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], candle_core::DType::F32, &dev) }.map_err(|x| e(x.to_string()))?;
        let model = BertModel::load(vb, &cfg).map_err(|x| e(x.to_string()))?;
        let name = format!("bert:{}", dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        Ok(BertEmbedder { model, tokenizer, dim, name })
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use candle_core::Tensor;
        let attempt = || -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
            let enc = self.tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| e.to_string())?;
            let ids: Vec<Vec<u32>> = enc.iter().map(|e| e.get_ids().to_vec()).collect();
            let mask: Vec<Vec<u32>> = enc.iter().map(|e| e.get_attention_mask().to_vec()).collect();
            let dev = candle_core::Device::Cpu;
            let ids = Tensor::new(ids, &dev)?;
            let mask = Tensor::new(mask, &dev)?;
            let types = ids.zeros_like()?;
            let out = self.model.forward(&ids, &types, Some(&mask))?;
            let cls = out.narrow(1, 0, 1)?.squeeze(1)?;
            let mut v = cls.to_vec2::<f32>()?;
            v.iter_mut().for_each(|x| normalize(x));
            Ok(v)
        };
        attempt().map_err(|e| model_err(format!("{}: {e}", self.name)))
    }
}

#[cfg(feature = "bert")]
impl Embedder for BertEmbedder {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Batches of similar length waste less on padding.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());
        let mut out = vec![Vec::new(); texts.len()];
        for chunk in order.chunks(16) {
            let batch: Vec<String> = chunk.iter().map(|&i| texts[i].to_string()).collect();
            for (&i, v) in chunk.iter().zip(self.run(&batch)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.run(&[format!("{}{text}", Self::QUERY_PREFIX)])?.pop().unwrap_or_default())
    }
}

// ---------------------------------------------------------------- siglip

/// SigLIP: a vision transformer and a text transformer trained so that an
/// image and its caption meet (`google/siglip-base-patch16-224`). Images
/// are resized to the model's square (bicubic, without keeping the aspect
/// ratio, as the model was trained) and scaled to [-1, 1]; texts are
/// lowercased without punctuation by the tokenizer, and padded to the 64
/// tokens the text tower pools the last of. Text is embedded alike for
/// documents and queries.
#[cfg(feature = "vision")]
pub struct SiglipEmbedder {
    model: candle_transformers::models::siglip::Model,
    tokenizer: tokenizers::Tokenizer,
    /// The side of the square images are resized to.
    side: usize,
    dim: usize,
    name: String,
}

#[cfg(feature = "vision")]
impl SiglipEmbedder {
    /// Images a forward pass takes at once.
    const BATCH: usize = 8;

    pub fn open(dir: &Path) -> Result<SiglipEmbedder> {
        use candle_nn::VarBuilder;
        use candle_transformers::models::siglip::{Config, Model};
        let e = |x: String| model_err(format!("{}: {x}", dir.display()));
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|x| e(x.to_string()))?;
        let cfg: Config = serde_json::from_str(&text).map_err(|x| e(x.to_string()))?;
        let len = cfg.text_config.max_position_embeddings;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|x| e(x.to_string()))?;
        tokenizer.with_truncation(Some(tokenizers::TruncationParams { max_length: len, ..Default::default() })).map_err(|x| e(x.to_string()))?;
        let pad = cfg.text_config.pad_token_id;
        tokenizer.with_padding(Some(tokenizers::PaddingParams { strategy: tokenizers::PaddingStrategy::Fixed(len), pad_id: pad, pad_token: "</s>".into(), ..Default::default() }));
        let dev = candle_core::Device::Cpu;
        // SAFETY: the weights file is not modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], candle_core::DType::F32, &dev) }.map_err(|x| e(x.to_string()))?;
        let model = Model::new(&cfg, vb).map_err(|x| e(x.to_string()))?;
        let name = format!("siglip:{}", dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        Ok(SiglipEmbedder { model, tokenizer, side: cfg.vision_config.image_size, dim: cfg.text_config.hidden_size, name })
    }

    fn texts(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let enc = self.tokenizer.encode_batch(texts.to_vec(), true)?;
        let ids: Vec<Vec<u32>> = enc.iter().map(|e| e.get_ids().to_vec()).collect();
        let ids = candle_core::Tensor::new(ids, &candle_core::Device::Cpu)?;
        let mut v = self.model.get_text_features(&ids)?.to_vec2::<f32>()?;
        v.iter_mut().for_each(|x| normalize(x));
        Ok(v)
    }

    /// The pixels the image tower takes: `side` by `side`, channels first,
    /// in [-1, 1].
    fn pixels(&self, image: &Image) -> Result<Vec<f32>> {
        let rgb = image::RgbImage::from_raw(image.width, image.height, image.rgb.clone()).ok_or_else(|| model_err("an image's pixels do not fill its size"))?;
        let side = self.side as u32;
        let small = image::imageops::resize(&rgb, side, side, image::imageops::FilterType::CatmullRom);
        let n = self.side * self.side;
        let mut out = vec![0f32; 3 * n];
        for (i, p) in small.pixels().enumerate() {
            for c in 0..3 {
                out[c * n + i] = p[c] as f32 / 127.5 - 1.0;
            }
        }
        Ok(out)
    }

    fn images(&self, images: &[&Image]) -> Result<Vec<Vec<f32>>> {
        let e = |x: candle_core::Error| model_err(format!("{}: {x}", self.name));
        let mut out = Vec::with_capacity(images.len());
        for batch in images.chunks(Self::BATCH) {
            let pixels = batch.iter().map(|i| self.pixels(i)).collect::<Result<Vec<_>>>()?.concat();
            let t = candle_core::Tensor::from_vec(pixels, (batch.len(), 3, self.side, self.side), &candle_core::Device::Cpu).map_err(e)?;
            let mut v = self.model.get_image_features(&t).and_then(|f| f.to_vec2::<f32>()).map_err(e)?;
            v.iter_mut().for_each(|x| normalize(x));
            out.extend(v);
        }
        Ok(out)
    }
}

#[cfg(feature = "vision")]
impl Embedder for SiglipEmbedder {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for c in texts.chunks(32) {
            out.extend(self.texts(c).map_err(|e| model_err(format!("{}: {e}", self.name)))?);
        }
        Ok(out)
    }
    fn modalities(&self) -> &'static [Modality] {
        &[Modality::Text, Modality::Image]
    }
    fn embed_inputs(&self, inputs: &[Input<'_>]) -> Result<Vec<Vec<f32>>> {
        if inputs.iter().any(|i| matches!(i, Input::Audio(_))) {
            return Err(model_err(format!("model {} does not embed sound", self.name)));
        }
        let texts: Vec<&str> = inputs.iter().filter_map(|i| if let Input::Text(t) = i { Some(*t) } else { None }).collect();
        let images: Vec<&Image> = inputs.iter().filter_map(|i| if let Input::Image(m) = i { Some(*m) } else { None }).collect();
        let (mut texts, mut images) = (self.embed(&texts)?.into_iter(), self.images(&images)?.into_iter());
        Ok(inputs.iter().map(|i| if let Input::Text(_) = i { texts.next() } else { images.next() }.unwrap_or_default()).collect())
    }
}
