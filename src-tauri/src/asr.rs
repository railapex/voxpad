// Two-pass ASR: Nemotron streaming + ParakeetTDT refinement.
//
// Thread 3: Nemotron — receives 560ms audio chunks, emits streaming text.
// Thread 4: TDT — receives complete utterances, emits refined text + word timestamps.
//
// Both models share the same ort CUDA context (initialized once via ort::init()).
// Natural GPU time-slicing: Nemotron runs during speech, TDT runs during silence.

use crate::rules;
use crossbeam_channel::Receiver;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use tauri::Emitter;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct NemotronState {
    model: parakeet_rs::Nemotron,
    #[allow(dead_code)]
    device: String,
}

struct TdtState {
    model: parakeet_rs::ParakeetTDT,
    #[allow(dead_code)]
    device: String,
}

// OnceLock<Result<...>> caches init errors so we don't retry (LinguaLens tts.rs pattern)
static NEMOTRON: OnceLock<Result<Mutex<NemotronState>, String>> = OnceLock::new();
static TDT: OnceLock<Result<Mutex<TdtState>, String>> = OnceLock::new();

// ---------------------------------------------------------------------------
// GPU cascade — CUDA → DirectML → CPU
// ---------------------------------------------------------------------------

fn gpu_cascade() -> Vec<(parakeet_rs::ExecutionConfig, &'static str)> {
    use parakeet_rs::{ExecutionConfig, ExecutionProvider};
    vec![
        (
            ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda),
            "CUDA",
        ),
        #[cfg(target_os = "windows")]
        (
            ExecutionConfig::new().with_execution_provider(ExecutionProvider::DirectML),
            "DirectML",
        ),
        (ExecutionConfig::new(), "CPU"),
    ]
}

fn init_with_cascade<T, F>(
    models_dir: &Path,
    name: &str,
    init_fn: F,
) -> Result<(T, String), String>
where
    F: Fn(&Path, parakeet_rs::ExecutionConfig) -> Result<T, Box<dyn std::error::Error>>,
{
    let providers = gpu_cascade();
    let mut last_err = String::new();

    for (config, provider_name) in providers {
        let t0 = std::time::Instant::now();
        match init_fn(models_dir, config) {
            Ok(model) => {
                log::info!(
                    "[asr] {} loaded on {} in {:.1?}",
                    name,
                    provider_name,
                    t0.elapsed()
                );
                return Ok((model, provider_name.to_string()));
            }
            Err(e) => {
                log::warn!("[asr] {} failed on {}: {}", name, provider_name, e);
                last_err = format!("{}: {}", provider_name, e);
            }
        }
    }

    Err(format!("[asr] {} init failed, all providers: {}", name, last_err))
}

// ---------------------------------------------------------------------------
// Init + warmup
// ---------------------------------------------------------------------------

pub fn init_nemotron(models_dir: &Path) -> Result<(), String> {
    let result = init_with_cascade(
        &models_dir.join("nemotron"),
        "Nemotron",
        |path, config| {
            let model = parakeet_rs::Nemotron::from_pretrained(path, Some(config))?;
            Ok(model)
        },
    );

    let result = match result {
        Ok((model, device)) => Ok(Mutex::new(NemotronState { model, device })),
        Err(e) => Err(e),
    };

    NEMOTRON.set(result).map_err(|_| "NEMOTRON already initialized".to_string())
}

pub fn init_tdt(models_dir: &Path) -> Result<(), String> {
    let result = init_with_cascade(
        &models_dir.join("tdt"),
        "TDT",
        |path, config| {
            let model = parakeet_rs::ParakeetTDT::from_pretrained(path, Some(config))?;
            Ok(model)
        },
    );

    let result = match result {
        Ok((model, device)) => Ok(Mutex::new(TdtState { model, device })),
        Err(e) => Err(e),
    };

    TDT.set(result).map_err(|_| "TDT already initialized".to_string())
}

pub fn warmup_nemotron() -> Result<(), String> {
    let state = NEMOTRON
        .get()
        .ok_or("NEMOTRON not initialized")?
        .as_ref()
        .map_err(|e| e.clone())?;

    let mut lock = state.lock().map_err(|e| format!("lock: {e}"))?;
    let silence = vec![0.0f32; 8960]; // 560ms
    let t0 = std::time::Instant::now();
    let _ = lock.model.transcribe_chunk(&silence);
    log::info!("[asr] Nemotron warmup in {:.1?}", t0.elapsed());
    Ok(())
}

pub fn warmup_tdt() -> Result<(), String> {
    use parakeet_rs::Transcriber;

    let state = TDT
        .get()
        .ok_or("TDT not initialized")?
        .as_ref()
        .map_err(|e| e.clone())?;

    let mut lock = state.lock().map_err(|e| format!("lock: {e}"))?;
    let silence = vec![0.0f32; 16000]; // 1 second
    let t0 = std::time::Instant::now();
    let _ = lock.model.transcribe_samples(silence, 16000, 1, None);
    log::info!("[asr] TDT warmup in {:.1?}", t0.elapsed());
    Ok(())
}

/// Preload both models on a background thread. Call at startup.
pub fn preload(models_dir: &Path) {
    log::info!("[asr] preloading models from {}", models_dir.display());

    if let Err(e) = init_nemotron(models_dir) {
        log::error!("[asr] Nemotron init: {e}");
    } else if let Err(e) = warmup_nemotron() {
        log::warn!("[asr] Nemotron warmup: {e}");
    }

    if let Err(e) = init_tdt(models_dir) {
        log::error!("[asr] TDT init: {e}");
    } else if let Err(e) = warmup_tdt() {
        log::warn!("[asr] TDT warmup: {e}");
    }
}

/// Check if models are loaded and ready.
pub fn is_ready() -> bool {
    NEMOTRON
        .get()
        .and_then(|r| r.as_ref().ok())
        .is_some()
        && TDT
            .get()
            .and_then(|r| r.as_ref().ok())
            .is_some()
}

// ---------------------------------------------------------------------------
// ASR event processing loop
// ---------------------------------------------------------------------------

/// Spawn the Nemotron streaming thread.
/// Receives 560ms audio chunks, produces streaming text.
pub fn spawn_nemotron_thread(
    rx: Receiver<Vec<f32>>,
    app_handle: tauri::AppHandle,
) {
    std::thread::Builder::new()
        .name("voxpad-nemotron".into())
        .spawn(move || {
            log::info!("[asr] nemotron thread started, waiting for chunks...");
            let mut chunk_count: u64 = 0;
            for chunk in rx {
                chunk_count += 1;
                log::debug!("[asr/nemotron] processing chunk #{} ({} samples)", chunk_count, chunk.len());
                process_nemotron_chunk(&chunk, chunk_count, &app_handle);
            }
            log::info!("[asr] nemotron thread ended after {} chunks", chunk_count);
        })
        .expect("spawn Nemotron thread");
}

/// Spawn the TDT refinement thread.
pub fn spawn_tdt_thread(
    rx: Receiver<Vec<f32>>,
    app_handle: tauri::AppHandle,
) {
    std::thread::Builder::new()
        .name("voxpad-tdt".into())
        .spawn(move || {
            log::info!("[asr] tdt thread started, waiting for utterances...");
            let mut utterance_id: u64 = 0;
            for audio in rx {
                utterance_id += 1;
                log::info!("[asr/tdt] received utterance #{} ({:.1}s)", utterance_id, audio.len() as f64 / 16000.0);
                process_tdt_utterance(audio, utterance_id, &app_handle);
            }
            log::info!("[asr] tdt thread ended after {} utterances", utterance_id);
        })
        .expect("spawn TDT thread");
}

fn process_nemotron_chunk(chunk: &[f32], chunk_id: u64, app: &tauri::AppHandle) {
    let state = match NEMOTRON.get().and_then(|r| r.as_ref().ok()) {
        Some(s) => s,
        None => {
            log::warn!("[asr/nemotron] model not loaded, skipping chunk #{}", chunk_id);
            return;
        }
    };

    let mut lock = state.lock().unwrap(); // dedicated thread, no contention

    let t0 = std::time::Instant::now();
    match lock.model.transcribe_chunk(chunk) {
        Ok(text) if !text.is_empty() => {
            let cleaned = rules::apply_t1(&text);
            if !cleaned.is_empty() {
                // Check if it's a command
                if let Some(cmd) = rules::detect_command(&cleaned) {
                    // Execute command on backend buffer
                    match cmd {
                        rules::BufferCommand::ScratchLast => {
                            crate::buffer::scratch_last();
                        }
                        rules::BufferCommand::ClearAll => {
                            crate::buffer::clear();
                        }
                    }
                    // Notify frontend to sync
                    app.emit(
                        "buffer-command",
                        serde_json::json!({
                            "command": format!("{:?}", cmd),
                            "text": crate::buffer::get_text(),
                        }),
                    )
                    .ok();
                } else {
                    log::debug!(
                        "[asr/nemotron] {:.0?}: '{}'",
                        t0.elapsed(),
                        &cleaned
                    );
                    // Update backend buffer state (for quick mode insertion)
                    crate::buffer::append_text(&cleaned);
                    // Update frontend display
                    app.emit(
                        "streaming-text",
                        serde_json::json!({
                            "text": cleaned,
                            "chunk_id": chunk_id,
                        }),
                    )
                    .ok();
                }
            }
        }
        Ok(_) => {} // empty result — normal for silence/noise
        Err(e) => log::warn!("[asr/nemotron] error: {e}"),
    }
}

fn process_tdt_utterance(audio: Vec<f32>, utterance_id: u64, app: &tauri::AppHandle) {
    use parakeet_rs::{TimestampMode, Transcriber};

    let state = match TDT.get().and_then(|r| r.as_ref().ok()) {
        Some(s) => s,
        None => {
            log::warn!("[asr/tdt] model not loaded, skipping utterance #{}", utterance_id);
            return;
        }
    };

    let mut lock = state.lock().unwrap(); // dedicated thread, no contention

    let duration_ms = (audio.len() as f64 / 16.0) as u64;
    let t0 = std::time::Instant::now();

    match lock
        .model
        .transcribe_samples(audio, 16000, 1, Some(TimestampMode::Words))
    {
        Ok(result) => {
            let cleaned = rules::apply_t1(&result.text);
            if !cleaned.is_empty() {
                log::info!(
                    "[asr/tdt] {:.0?}: '{}' ({}ms audio)",
                    t0.elapsed(),
                    &cleaned,
                    duration_ms
                );

                // Update backend buffer (for quick mode insertion)
                crate::buffer::append_text(&cleaned);

                let words: Vec<_> = result
                    .tokens
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                        })
                    })
                    .collect();

                // Send to frontend for display
                app.emit(
                    "refined-text",
                    serde_json::json!({
                        "text": cleaned,
                        "words": words,
                        "utterance_id": utterance_id,
                        "duration_ms": duration_ms,
                    }),
                )
                .ok();
            }
        }
        Err(e) => log::warn!("[asr/tdt] error: {e}"),
    }
}
