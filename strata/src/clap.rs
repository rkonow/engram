//! CLAP (feature `audio`): sound and text in one space, as SigLIP puts
//! images and text (`laion/clap-htsat-unfused`, Apache-2.0). The audio
//! tower is HTSAT, a Swin transformer over a log-mel spectrogram folded
//! into a square; the text tower is RoBERTa. Both are Strata's own, in
//! candle, over the repository's `pytorch_model.bin`.

use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{conv2d, embedding, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder};

use crate::audio::gelu;
use crate::embed::{Audio, Embedder, Input, Modality};
use crate::error::{Error, Result};

type CResult<T> = candle_core::Result<T>;

fn model_err(what: impl std::fmt::Display) -> Error {
    Error::Schema(what.to_string())
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// CLAP's audio and text towers. Sound is resampled to 48 kHz and read
/// in 10-second windows, each a log-mel spectrogram as the
/// `transformers` library computes it for this model (64 Slaney mel bins
/// from 50 Hz to 14 kHz, in decibels), a window shorter than 10 seconds
/// repeated to fill it and padded with silence. A sound's vector is the
/// mean of its windows' vectors, normalized: `transformers` takes one
/// random 10 seconds of a longer sound instead. Texts are embedded alike
/// for documents and queries.
pub struct ClapEmbedder {
    audio: Htsat,
    audio_projection: Projection,
    text: Roberta,
    text_projection: Projection,
    tokenizer: tokenizers::Tokenizer,
    /// `(64, 513)`.
    filters: Tensor,
    /// Bicubic resizing of the spectrogram's 1001 frames to the 1024 the
    /// audio tower folds: `(1024, 1001)`.
    stretch: Tensor,
    dim: usize,
    name: String,
}

impl ClapEmbedder {
    const RATE: u32 = 48_000;
    /// Samples a window holds: 10 seconds.
    const WINDOW: usize = 480_000;
    const HOP: usize = 480;
    const N_FFT: usize = 1024;
    /// Windows a forward pass takes at once.
    const BATCH: usize = 8;

    pub fn open(dir: &Path) -> Result<ClapEmbedder> {
        let e = |x: String| model_err(format!("{}: {x}", dir.display()));
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|x| e(x.to_string()))?;
        let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|x| e(x.to_string()))?;
        let audio_cfg: AudioConfig = serde_json::from_value(cfg["audio_config"].clone()).map_err(|x| e(format!("audio_config: {x}")))?;
        let text_cfg: TextConfig = serde_json::from_value(cfg["text_config"].clone()).map_err(|x| e(format!("text_config: {x}")))?;
        let dim = cfg["projection_dim"].as_u64().unwrap_or(512) as usize;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|x| e(x.to_string()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams { max_length: text_cfg.max_position_embeddings - text_cfg.pad_token_id as usize - 1, ..Default::default() }))
            .map_err(|x| e(x.to_string()))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams { pad_id: text_cfg.pad_token_id, pad_token: "<pad>".into(), ..Default::default() }));
        let dev = Device::Cpu;
        let safetensors = dir.join("model.safetensors");
        let tensors: std::collections::HashMap<String, Tensor> = if safetensors.exists() {
            candle_core::safetensors::load(&safetensors, &dev)
        } else {
            candle_core::pickle::read_all(dir.join("pytorch_model.bin")).map(|t| t.into_iter().collect())
        }
        .map_err(|x| e(x.to_string()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let load = || -> CResult<(Htsat, Projection, Roberta, Projection)> {
            Ok((
                Htsat::new(&audio_cfg, vb.pp("audio_model.audio_encoder"))?,
                Projection::new(audio_cfg.hidden_size, dim, vb.pp("audio_projection"))?,
                Roberta::new(&text_cfg, vb.pp("text_model"))?,
                Projection::new(text_cfg.hidden_size, dim, vb.pp("text_projection"))?,
            ))
        };
        let (audio, audio_projection, text, text_projection) = load().map_err(|x| e(x.to_string()))?;
        let filters = crate::audio::mel_filters(Self::RATE, Self::N_FFT, audio_cfg.num_mel_bins, 50.0, 14_000.0, false, true);
        let filters = Tensor::from_vec(filters.concat(), (audio_cfg.num_mel_bins, Self::N_FFT / 2 + 1), &dev).map_err(|x| e(x.to_string()))?;
        let frames = Self::WINDOW / Self::HOP + 1;
        let folded = audio_cfg.spec_size * (audio_cfg.spec_size / audio_cfg.num_mel_bins);
        let stretch = Tensor::from_vec(bicubic(frames, folded), (folded, frames), &dev).map_err(|x| e(x.to_string()))?;
        let name = format!("clap:{}", dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        Ok(ClapEmbedder { audio, audio_projection, text, text_projection, tokenizer, filters, stretch, dim, name })
    }

    fn texts(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let enc = self.tokenizer.encode_batch(texts.to_vec(), true)?;
        let ids: Vec<Vec<u32>> = enc.iter().map(|e| e.get_ids().to_vec()).collect();
        let mask: Vec<Vec<u32>> = enc.iter().map(|e| e.get_attention_mask().to_vec()).collect();
        let dev = Device::Cpu;
        let pooled = self.text.forward(&Tensor::new(ids, &dev)?, &Tensor::new(mask, &dev)?)?;
        let mut v = self.text_projection.forward(&pooled)?.to_vec2::<f32>()?;
        v.iter_mut().for_each(|x| normalize(x));
        Ok(v)
    }

    /// The log-mel spectrogram of a window of 48 kHz sound, `(1001, 64)`:
    /// the window repeated to fill 10 seconds and padded with silence,
    /// frames 10 ms apart, the mel power in decibels.
    fn mel(&self, window: &[f32]) -> CResult<Tensor> {
        let mut x = Vec::with_capacity(Self::WINDOW);
        if !window.is_empty() {
            for _ in 0..(Self::WINDOW / window.len()) {
                x.extend_from_slice(window);
            }
        }
        x.resize(Self::WINDOW, 0.0);
        let power = crate::audio::power_spectrogram(&x, Self::N_FFT, Self::HOP, self.filters.device())?;
        let mel = power.matmul(&self.filters.t()?)?;
        mel.maximum(1e-10)?.log()? * (10.0 / std::f64::consts::LN_10)
    }

    /// The vectors of windows of 48 kHz sound, normalized.
    fn windows(&self, windows: &[&[f32]]) -> CResult<Vec<Vec<f32>>> {
        let mels = windows.iter().map(|w| self.mel(w)).collect::<CResult<Vec<_>>>()?;
        let mel = self.stretch.broadcast_matmul(&Tensor::stack(&mels, 0)?)?;
        let mut v = self.audio_projection.forward(&self.audio.forward(&mel)?)?.to_vec2::<f32>()?;
        v.iter_mut().for_each(|x| normalize(x));
        Ok(v)
    }

    fn sounds(&self, sounds: &[&Audio]) -> Result<Vec<Vec<f32>>> {
        let pcm: Vec<Vec<f32>> = sounds.iter().map(|a| crate::audio::resample(&a.samples, a.rate, Self::RATE)).collect();
        let windows: Vec<(usize, &[f32])> = pcm.iter().enumerate().flat_map(|(i, x)| x.chunks(Self::WINDOW).map(move |w| (i, w))).collect();
        let mut sums = vec![vec![0f32; self.dim]; sounds.len()];
        for batch in windows.chunks(Self::BATCH) {
            let v = self.windows(&batch.iter().map(|(_, w)| *w).collect::<Vec<_>>()).map_err(|x| model_err(format!("{}: {x}", self.name)))?;
            for ((i, _), v) in batch.iter().zip(v) {
                sums[*i].iter_mut().zip(v).for_each(|(s, x)| *s += x);
            }
        }
        sums.iter_mut().for_each(|s| normalize(s));
        Ok(sums)
    }
}

impl Embedder for ClapEmbedder {
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
        &[Modality::Text, Modality::Audio]
    }
    fn embed_inputs(&self, inputs: &[Input<'_>]) -> Result<Vec<Vec<f32>>> {
        if inputs.iter().any(|i| matches!(i, Input::Image(_))) {
            return Err(model_err(format!("model {} does not embed images", self.name)));
        }
        let texts: Vec<&str> = inputs.iter().filter_map(|i| if let Input::Text(t) = i { Some(*t) } else { None }).collect();
        let sounds: Vec<&Audio> = inputs.iter().filter_map(|i| if let Input::Audio(a) = i { Some(*a) } else { None }).collect();
        let (mut texts, mut sounds) = (self.embed(&texts)?.into_iter(), self.sounds(&sounds)?.into_iter());
        Ok(inputs.iter().map(|i| if let Input::Text(_) = i { texts.next() } else { sounds.next() }.unwrap_or_default()).collect())
    }
}

/// The weights that resize `from` samples to `to` by bicubic
/// interpolation with the ends aligned, as PyTorch's `interpolate(...,
/// mode="bicubic", align_corners=True)` does (a = -0.75, the edge sample
/// repeated past the ends): `to` rows of `from`.
fn bicubic(from: usize, to: usize) -> Vec<f32> {
    let a = -0.75f64;
    let near = |x: f64| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let far = |x: f64| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    let scale = if to > 1 { (from - 1) as f64 / (to - 1) as f64 } else { 0.0 };
    let mut w = vec![0f32; to * from];
    for i in 0..to {
        let src = i as f64 * scale;
        let at = src.floor();
        let t = src - at;
        let k = [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)];
        for (j, k) in k.iter().enumerate() {
            let idx = (at as isize - 1 + j as isize).clamp(0, from as isize - 1) as usize;
            w[i * from + idx] += *k as f32;
        }
    }
    w
}

/// Two layers from a tower's pooled state to the shared space.
struct Projection {
    linear1: Linear,
    linear2: Linear,
}

impl Projection {
    fn new(from: usize, dim: usize, vb: VarBuilder) -> CResult<Projection> {
        Ok(Projection { linear1: linear(from, dim, vb.pp("linear1"))?, linear2: linear(dim, dim, vb.pp("linear2"))? })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        self.linear2.forward(&self.linear1.forward(x)?.relu()?)
    }
}

// ---------------------------------------------------------------- audio

/// What of `audio_config` HTSAT needs.
#[derive(serde::Deserialize)]
struct AudioConfig {
    depths: Vec<usize>,
    num_attention_heads: Vec<usize>,
    hidden_size: usize,
    patch_embeds_hidden_size: usize,
    patch_size: usize,
    num_mel_bins: usize,
    spec_size: usize,
    window_size: usize,
    #[serde(default)]
    enable_fusion: bool,
}

/// HTSAT: the spectrogram, normalized per mel bin, its 1024 frames folded
/// into four bands of 256 one above the other (a 256 by 256 image),
/// patches of 4 by 4, then Swin stages of shifted-window attention, each
/// but the last merging 2 by 2 patches; the vector is the mean of the
/// last stage's patches.
struct Htsat {
    bn_scale: Tensor,
    bn_shift: Tensor,
    patch: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
    norm: LayerNorm,
    spec_size: usize,
    mels: usize,
}

impl Htsat {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> CResult<Htsat> {
        if cfg.enable_fusion {
            candle_core::bail!("CLAP with feature fusion is not supported (use an unfused model)");
        }
        let bn = vb.pp("batch_norm");
        let var = bn.get(cfg.num_mel_bins, "running_var")?;
        let bn_scale = (bn.get(cfg.num_mel_bins, "weight")? / (var + 1e-5)?.sqrt()?)?;
        let bn_shift = (bn.get(cfg.num_mel_bins, "bias")? - (bn.get(cfg.num_mel_bins, "running_mean")? * &bn_scale)?)?;
        let patch = conv2d(1, cfg.patch_embeds_hidden_size, cfg.patch_size, Conv2dConfig { stride: cfg.patch_size, ..Default::default() }, vb.pp("patch_embed.proj"))?;
        let patch_norm = layer_norm(cfg.patch_embeds_hidden_size, 1e-5, vb.pp("patch_embed.norm"))?;
        let mut stages = Vec::new();
        let mut side = cfg.spec_size / cfg.patch_size;
        for (i, (&depth, &heads)) in cfg.depths.iter().zip(&cfg.num_attention_heads).enumerate() {
            let dim = cfg.patch_embeds_hidden_size << i;
            let last = i + 1 == cfg.depths.len();
            stages.push(Stage::new(dim, heads, depth, side, cfg.window_size, !last, vb.pp("layers").pp(i))?);
            side /= 2;
        }
        let norm = layer_norm(cfg.hidden_size, 1e-5, vb.pp("norm"))?;
        Ok(Htsat { bn_scale, bn_shift, patch, patch_norm, stages, norm, spec_size: cfg.spec_size, mels: cfg.num_mel_bins })
    }

    /// The pooled state `(batch, hidden)` of spectrograms `(batch, 1024,
    /// mel bins)`, their frames already resized.
    fn forward(&self, mel: &Tensor) -> CResult<Tensor> {
        let b = mel.dim(0)?;
        let x = mel.broadcast_mul(&self.bn_scale)?.broadcast_add(&self.bn_shift)?;
        let bands = self.spec_size / self.mels;
        let x = x.reshape((b, bands, self.spec_size, self.mels))?.transpose(2, 3)?.contiguous()?.reshape((b, 1, self.spec_size, self.spec_size))?;
        let x = self.patch.forward(&x)?;
        let mut x = self.patch_norm.forward(&x.flatten_from(2)?.transpose(1, 2)?)?;
        for s in &self.stages {
            x = s.forward(&x)?;
        }
        self.norm.forward(&x)?.mean(1)
    }
}

/// A Swin stage: blocks of window attention, every second one over
/// windows shifted by half a window, and a merge of 2 by 2 patches.
struct Stage {
    blocks: Vec<Block>,
    merge: Option<(LayerNorm, Linear)>,
    side: usize,
}

impl Stage {
    fn new(dim: usize, heads: usize, depth: usize, side: usize, window: usize, merge: bool, vb: VarBuilder) -> CResult<Stage> {
        // A stage no larger than a window attends to all of it, unshifted.
        let window = window.min(side);
        let blocks = (0..depth)
            .map(|i| {
                let shift = if i % 2 == 1 && side > window { window / 2 } else { 0 };
                Block::new(dim, heads, side, window, shift, vb.pp("blocks").pp(i))
            })
            .collect::<CResult<_>>()?;
        let merge = match merge {
            true => Some((layer_norm(4 * dim, 1e-5, vb.pp("downsample.norm"))?, linear_no_bias(4 * dim, 2 * dim, vb.pp("downsample.reduction"))?)),
            false => None,
        };
        Ok(Stage { blocks, merge, side })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let Some((norm, reduction)) = &self.merge else { return Ok(x) };
        // Each 2 by 2 patch's four, in the order (0, 0), (1, 0), (0, 1),
        // (1, 1) of (row, column), side by side.
        let (b, _, c) = x.dims3()?;
        let h = self.side / 2;
        let x = x.reshape((b, h, 2, h, 2, c))?.permute((0, 1, 3, 4, 2, 5))?.contiguous()?.reshape((b, h * h, 4 * c))?;
        reduction.forward(&norm.forward(&x)?)
    }
}

struct Block {
    norm1: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    /// `(heads, window², window²)`, with the shifted windows' mask
    /// `(windows, window², window²)` if shifted.
    bias: Tensor,
    mask: Option<Tensor>,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    side: usize,
    window: usize,
    shift: usize,
}

impl Block {
    fn new(dim: usize, heads: usize, side: usize, window: usize, shift: usize, vb: VarBuilder) -> CResult<Block> {
        let att = vb.pp("attention");
        let table = att.get(((2 * window - 1) * (2 * window - 1), heads), "self.relative_position_bias_table")?;
        let n = window * window;
        let mut index = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let (dy, dx) = ((i / window) as isize - (j / window) as isize, (i % window) as isize - (j % window) as isize);
                index.push(((dy + window as isize - 1) * (2 * window as isize - 1) + dx + window as isize - 1) as u32);
            }
        }
        let bias = table.index_select(&Tensor::new(index, vb.device())?, 0)?.reshape((n, n, heads))?.permute((2, 0, 1))?.contiguous()?;
        let mask = match shift {
            0 => None,
            s => Some(shifted_mask(side, window, s, vb.device())?),
        };
        Ok(Block {
            norm1: layer_norm(dim, 1e-5, vb.pp("layernorm_before"))?,
            query: linear(dim, dim, att.pp("self.query"))?,
            key: linear(dim, dim, att.pp("self.key"))?,
            value: linear(dim, dim, att.pp("self.value"))?,
            out: linear(dim, dim, att.pp("output.dense"))?,
            bias,
            mask,
            norm2: layer_norm(dim, 1e-5, vb.pp("layernorm_after"))?,
            fc1: linear(dim, 4 * dim, vb.pp("intermediate.dense"))?,
            fc2: linear(4 * dim, dim, vb.pp("output.dense"))?,
            heads,
            side,
            window,
            shift,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, _, c) = x.dims3()?;
        let (s, w) = (self.side, self.window);
        let mut h = self.norm1.forward(x)?.reshape((b, s, s, c))?;
        if self.shift > 0 {
            h = h.roll(-(self.shift as i32), 1)?.roll(-(self.shift as i32), 2)?;
        }
        // Windows: (batch × windows, window², channels).
        let k = s / w;
        let win = h.reshape((b, k, w, k, w, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b * k * k, w * w, c))?;
        let att = self.attend(&win, b)?;
        let mut h = att.reshape((b, k, k, w, w, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, s, s, c))?;
        if self.shift > 0 {
            h = h.roll(self.shift as i32, 1)?.roll(self.shift as i32, 2)?;
        }
        let x = (x + h.reshape((b, s * s, c))?)?;
        &x + self.fc2.forward(&gelu(&self.fc1.forward(&self.norm2.forward(&x)?)?)?)?
    }

    /// Attention within each window of `x` `(batch × windows, window²,
    /// channels)`.
    fn attend(&self, x: &Tensor, batch: usize) -> CResult<Tensor> {
        let (nw, n, c) = x.dims3()?;
        let dh = c / self.heads;
        let split = |t: Tensor| t.reshape((nw, n, self.heads, dh))?.transpose(1, 2)?.contiguous();
        let q = (split(self.query.forward(x)?)? * (1.0 / (dh as f64).sqrt()))?;
        let (k, v) = (split(self.key.forward(x)?)?, split(self.value.forward(x)?)?);
        let mut scores = q.matmul(&k.t()?)?.broadcast_add(&self.bias)?;
        if let Some(m) = &self.mask {
            let windows = nw / batch;
            scores = scores.reshape((batch, windows, self.heads, n, n))?.broadcast_add(&m.unsqueeze(1)?.unsqueeze(0)?)?.reshape((nw, self.heads, n, n))?;
        }
        let w = candle_nn::ops::softmax_last_dim(&scores)?;
        self.out.forward(&w.matmul(&v)?.transpose(1, 2)?.reshape((nw, n, c))?)
    }
}

/// The mask of windows shifted by `shift` over a `side` by `side` grid:
/// `-100` between two patches that the shift brought from different
/// parts of the grid, `(windows, window², window²)`.
fn shifted_mask(side: usize, window: usize, shift: usize, dev: &Device) -> CResult<Tensor> {
    let part = |i: usize| (i >= side - window) as usize + (i >= side - shift) as usize;
    let k = side / window;
    let n = window * window;
    let mut m = Vec::with_capacity(k * k * n * n);
    for wy in 0..k {
        for wx in 0..k {
            let label: Vec<usize> = (0..n).map(|p| part(wy * window + p / window) * 3 + part(wx * window + p % window)).collect();
            for i in 0..n {
                for j in 0..n {
                    m.push(if label[i] == label[j] { 0f32 } else { -100.0 });
                }
            }
        }
    }
    Tensor::from_vec(m, (k * k, n, n), dev)
}

// ---------------------------------------------------------------- text

/// What of `text_config` RoBERTa needs.
#[derive(serde::Deserialize)]
struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    pad_token_id: u32,
    layer_norm_eps: f64,
}

/// RoBERTa: BERT's post-norm encoder with positions counted from the
/// padding token's id plus one; pooled by the first token's state
/// through a dense layer and tanh.
struct Roberta {
    words: Embedding,
    positions: Embedding,
    token_type: Tensor,
    norm: LayerNorm,
    layers: Vec<TextLayer>,
    pooler: Linear,
    pad: u32,
}

struct TextLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    norm1: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    norm2: LayerNorm,
    heads: usize,
}

impl Roberta {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> CResult<Roberta> {
        let (d, eps) = (cfg.hidden_size, cfg.layer_norm_eps);
        let emb = vb.pp("embeddings");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let l = vb.pp("encoder.layer").pp(i);
                Ok(TextLayer {
                    query: linear(d, d, l.pp("attention.self.query"))?,
                    key: linear(d, d, l.pp("attention.self.key"))?,
                    value: linear(d, d, l.pp("attention.self.value"))?,
                    out: linear(d, d, l.pp("attention.output.dense"))?,
                    norm1: layer_norm(d, eps, l.pp("attention.output.LayerNorm"))?,
                    fc1: linear(d, cfg.intermediate_size, l.pp("intermediate.dense"))?,
                    fc2: linear(cfg.intermediate_size, d, l.pp("output.dense"))?,
                    norm2: layer_norm(d, eps, l.pp("output.LayerNorm"))?,
                    heads: cfg.num_attention_heads,
                })
            })
            .collect::<CResult<_>>()?;
        Ok(Roberta {
            words: embedding(cfg.vocab_size, d, emb.pp("word_embeddings"))?,
            positions: embedding(cfg.max_position_embeddings, d, emb.pp("position_embeddings"))?,
            token_type: emb.get((1, d), "token_type_embeddings.weight")?,
            norm: layer_norm(d, eps, emb.pp("LayerNorm"))?,
            layers,
            pooler: linear(d, d, vb.pp("pooler.dense"))?,
            pad: cfg.pad_token_id,
        })
    }

    /// The pooled state `(batch, hidden)` of token ids `(batch, len)`
    /// with their attention mask (1 for a token, 0 for padding).
    fn forward(&self, ids: &Tensor, mask: &Tensor) -> CResult<Tensor> {
        // A token's position is the padding id plus its count among the
        // tokens so far; padding's is the padding id.
        let m: Vec<Vec<u32>> = mask.to_vec2()?;
        let pos: Vec<Vec<u32>> = m
            .iter()
            .map(|row| {
                let mut n = 0;
                row.iter().map(|&t| if t == 1 { n += 1; self.pad + n } else { self.pad }).collect()
            })
            .collect();
        let pos = Tensor::new(pos, ids.device())?;
        let x = self.words.forward(ids)?.add(&self.positions.forward(&pos)?)?.broadcast_add(&self.token_type)?;
        let mut x = self.norm.forward(&x)?;
        let (b, n) = ids.dims2()?;
        let keep = mask.to_dtype(DType::F32)?.reshape((b, 1, 1, n))?;
        let bias = ((1.0 - keep)? * f32::MIN as f64)?;
        for l in &self.layers {
            let (_, _, d) = x.dims3()?;
            let dh = d / l.heads;
            let split = |t: Tensor| t.reshape((b, n, l.heads, dh))?.transpose(1, 2)?.contiguous();
            let q = (split(l.query.forward(&x)?)? * (1.0 / (dh as f64).sqrt()))?;
            let (k, v) = (split(l.key.forward(&x)?)?, split(l.value.forward(&x)?)?);
            let w = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?.broadcast_add(&bias)?)?;
            let ctx = w.matmul(&v)?.transpose(1, 2)?.reshape((b, n, d))?;
            let a = l.norm1.forward(&(l.out.forward(&ctx)? + &x)?)?;
            x = l.norm2.forward(&(l.fc2.forward(&gelu(&l.fc1.forward(&a)?)?)? + &a)?)?;
        }
        self.pooler.forward(&x.narrow(1, 0, 1)?.squeeze(1)?)?.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_keeps_the_samples_and_a_line_inside() {
        let w = bicubic(5, 9);
        let x = [0f32, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = (0..9).map(|i| (0..5).map(|j| w[i * 5 + j] * x[j]).sum()).collect();
        // The samples themselves (every other point), and a line between
        // them where the four neighbours are all inside: next to an end,
        // the end is repeated, as PyTorch's bicubic does.
        for i in [0, 2, 3, 4, 5, 6, 8] {
            assert!((y[i] - i as f32 * 0.5).abs() < 1e-5, "{i}: {}", y[i]);
        }
        assert!(y[1] > 0.0 && y[1] < 0.5 && y[7] > 3.5 && y[7] < 4.0, "{y:?}");
        // Each row's weights add up to one.
        assert!((0..9).all(|i| ((0..5).map(|j| w[i * 5 + j]).sum::<f32>() - 1.0).abs() < 1e-5));
    }

    #[test]
    fn a_shifted_window_masks_what_the_shift_brought_together() {
        // 4 by 4 patches, windows of 2, shift 1: the last window holds
        // patches from four parts of the grid.
        let m = shifted_mask(4, 2, 1, &Device::Cpu).unwrap();
        assert_eq!(m.dims(), [4, 4, 4]);
        let first: Vec<Vec<f32>> = m.get(0).unwrap().to_vec2().unwrap();
        assert!(first.iter().flatten().all(|&x| x == 0.0));
        let last: Vec<Vec<f32>> = m.get(3).unwrap().to_vec2().unwrap();
        assert!((0..4).all(|i| last[i][i] == 0.0) && (0..4).all(|i| (0..4).filter(|&j| j != i).all(|j| last[i][j] == -100.0)));
    }
}
