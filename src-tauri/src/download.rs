// Model download with streaming progress and resume support.
// Adapted from LinguaLens D:/dev/lingualens/src-tauri/src/download.rs

use futures_util::StreamExt;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::Emitter;

static CANCEL: AtomicBool = AtomicBool::new(false);

const HF_RESOLVE: &str = "https://huggingface.co";

struct ModelFile {
    name: &'static str,
    repo: &'static str,
    path: &'static str,
    dest: &'static str,
    size_bytes: u64,
}

// Model manifest — HuggingFace repos verified 2026-03-28
const DOWNLOADS: &[ModelFile] = &[
    // Nemotron Speech Streaming 0.6B — streaming ASR with punctuation
    ModelFile {
        name: "Nemotron encoder",
        repo: "altunenes/parakeet-rs",
        path: "nemotron-speech-streaming-en-0.6b/encoder.onnx",
        dest: "nemotron/encoder.onnx",
        size_bytes: 1_000_000, // small metadata, actual weights in .data
    },
    ModelFile {
        name: "Nemotron encoder weights",
        repo: "altunenes/parakeet-rs",
        path: "nemotron-speech-streaming-en-0.6b/encoder.onnx.data",
        dest: "nemotron/encoder.onnx.data",
        size_bytes: 2_436_567_040, // ~2.4GB
    },
    ModelFile {
        name: "Nemotron decoder",
        repo: "altunenes/parakeet-rs",
        path: "nemotron-speech-streaming-en-0.6b/decoder_joint.onnx",
        dest: "nemotron/decoder_joint.onnx",
        size_bytes: 50_000_000, // ~50MB estimated
    },
    ModelFile {
        name: "Nemotron tokenizer",
        repo: "altunenes/parakeet-rs",
        path: "nemotron-speech-streaming-en-0.6b/tokenizer.model",
        dest: "nemotron/tokenizer.model",
        size_bytes: 500_000, // ~500KB
    },
    // Parakeet TDT 0.6B v3 — batch refinement with word timestamps
    ModelFile {
        name: "TDT encoder",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        path: "encoder-model.onnx",
        dest: "tdt/encoder-model.onnx",
        size_bytes: 1_000_000,
    },
    ModelFile {
        name: "TDT encoder weights",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        path: "encoder-model.onnx.data",
        dest: "tdt/encoder-model.onnx.data",
        size_bytes: 2_435_420_160, // ~2.4GB
    },
    ModelFile {
        name: "TDT decoder",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        path: "decoder_joint-model.onnx",
        dest: "tdt/decoder_joint-model.onnx",
        size_bytes: 50_000_000,
    },
    ModelFile {
        name: "TDT vocab",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        path: "vocab.txt",
        dest: "tdt/vocab.txt",
        size_bytes: 10_000,
    },
    // Silero VAD — voice activity detection
    ModelFile {
        name: "Silero VAD",
        repo: "onnx-community/silero-vad",
        path: "onnx/model.onnx",
        dest: "silero_vad.onnx",
        size_bytes: 2_243_022, // ~2.2MB
    },
];

#[derive(Serialize, Clone)]
pub struct MissingModel {
    pub name: String,
    pub size_bytes: u64,
}

#[derive(Serialize, Clone)]
pub struct DownloadProgress {
    pub name: String,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub overall_bytes_downloaded: u64,
    pub overall_bytes_total: u64,
}

/// Check which models are missing.
pub fn check_models(data_dir: &Path) -> Vec<MissingModel> {
    let models_dir = data_dir.join("models");
    DOWNLOADS
        .iter()
        .filter(|m| !models_dir.join(m.dest).exists())
        .map(|m| MissingModel {
            name: m.name.to_string(),
            size_bytes: m.size_bytes,
        })
        .collect()
}

/// Download all missing models with progress events and resume support.
pub async fn download_models(
    data_dir: PathBuf,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    CANCEL.store(false, Ordering::SeqCst);

    let models_dir = data_dir.join("models");
    let missing: Vec<&ModelFile> = DOWNLOADS
        .iter()
        .filter(|m| !models_dir.join(m.dest).exists())
        .collect();

    if missing.is_empty() {
        log::info!("[download] all models present");
        return Ok(());
    }

    let overall_total: u64 = missing.iter().map(|m| m.size_bytes).sum();
    let mut overall_downloaded: u64 = missing
        .iter()
        .map(|mf| {
            let partial = models_dir.join(format!("{}.partial", mf.dest));
            std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0)
        })
        .sum();

    log::info!(
        "[download] {} files to download ({:.1} GB total)",
        missing.len(),
        overall_total as f64 / 1e9
    );

    let client = reqwest::Client::new();

    for model in &missing {
        if CANCEL.load(Ordering::SeqCst) {
            return Err("Download cancelled".into());
        }

        let dest = models_dir.join(model.dest);
        let partial = models_dir.join(format!("{}.partial", model.dest));

        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("mkdir: {e}"))?;
        }

        let url = format!("{}/{}/resolve/main/{}", HF_RESOLVE, model.repo, model.path);

        // Resume support
        let existing_bytes = tokio::fs::metadata(&partial)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        let mut request = client.get(&url);
        if existing_bytes > 0 {
            request = request.header("Range", format!("bytes={}-", existing_bytes));
            log::info!("[download] resuming {} from {} bytes", model.name, existing_bytes);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP request failed for {}: {e}", model.name))?;

        if !response.status().is_success() && response.status().as_u16() != 206 {
            return Err(format!("HTTP {} for {}", response.status(), model.name));
        }

        let content_length = response.content_length().unwrap_or(0);
        let total_size = if response.status().as_u16() == 206 {
            existing_bytes + content_length
        } else {
            content_length
        };

        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(existing_bytes > 0 && response.status().as_u16() == 206)
            .write(true)
            .truncate(existing_bytes == 0 || response.status().as_u16() != 206)
            .open(&partial)
            .await
            .map_err(|e| format!("open {}: {e}", partial.display()))?;

        let mut stream = response.bytes_stream();
        let mut downloaded = existing_bytes;

        while let Some(chunk) = stream.next().await {
            if CANCEL.load(Ordering::SeqCst) {
                return Err("Download cancelled".into());
            }

            let chunk = chunk.map_err(|e| format!("download error for {}: {e}", model.name))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("write error: {e}"))?;

            downloaded += chunk.len() as u64;
            overall_downloaded += chunk.len() as u64;

            // Progress every ~100KB
            if downloaded % (100 * 1024) < chunk.len() as u64 || downloaded >= total_size {
                let _ = app_handle.emit(
                    "download-progress",
                    DownloadProgress {
                        name: model.name.to_string(),
                        bytes_downloaded: downloaded,
                        bytes_total: total_size,
                        overall_bytes_downloaded: overall_downloaded,
                        overall_bytes_total: overall_total,
                    },
                );
            }
        }

        file.flush().await.map_err(|e| format!("flush: {e}"))?;
        drop(file);

        // Validate size
        let actual = tokio::fs::metadata(&partial)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if total_size > 0 && actual != total_size {
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(format!(
                "{}: expected {} bytes, got {} — removed, retry",
                model.name, total_size, actual
            ));
        }

        // Atomic rename
        tokio::fs::rename(&partial, &dest)
            .await
            .map_err(|e| format!("rename: {e}"))?;

        log::info!("[download] {} complete ({} bytes)", model.name, downloaded);
    }

    let _ = app_handle.emit("download-complete", ());
    Ok(())
}

/// Cancel an in-progress download.
pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}
