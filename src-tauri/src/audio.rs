// Audio capture pipeline — cpal mic capture, rubato resample, VAD dispatch.
//
// Architecture:
//   cpal callback thread (OS high-priority)
//     → crossbeam bounded channel (try_send, drops if full)
//     → Processing thread (dedicated, long-lived)
//       → rubato resample native_rate → 16kHz
//       → Silero VAD per 512-sample frame (32ms)
//       → Speech: feed 8960-sample chunks (560ms) to Nemotron channel
//       → Silence: send accumulated utterance to TDT channel

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tauri::Emitter;
use cpal::{SampleFormat, SampleRate, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::vad::{SileroVad, VadEvent};

const TARGET_SAMPLE_RATE: u32 = 16000;
const NEMOTRON_CHUNK_SAMPLES: usize = 8960; // 560ms at 16kHz
const VAD_FRAME_SAMPLES: usize = 512; // 32ms at 16kHz

/// Messages sent from the audio pipeline to consumers.
#[derive(Debug)]
pub enum AudioEvent {
    /// 560ms chunk for Nemotron streaming ASR
    NemotronChunk(Vec<f32>),
    /// Complete utterance for TDT refinement
    TdtUtterance(Vec<f32>),
    /// VAD state change
    VadEvent(VadEvent),
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Handle for a running capture session. Drop to stop.
pub struct CaptureHandle {
    _stream: cpal::Stream, // kept alive — dropping stops capture
    stop: Arc<AtomicBool>,
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        log::info!("[audio] capture stopped");
    }
}

/// List available input devices.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    match host.input_devices() {
        Ok(devices) => devices
            .filter_map(|d| {
                let name = d.name().ok()?;
                Some(DeviceInfo {
                    is_default: name == default_name,
                    name,
                })
            })
            .collect(),
        Err(e) => {
            log::error!("[audio] failed to enumerate devices: {e}");
            vec![]
        }
    }
}

/// Start capturing audio from the specified device (or default).
/// Returns a CaptureHandle (drop to stop) and spawns the processing thread.
///
/// `event_tx` receives AudioEvents (Nemotron chunks, TDT utterances, VAD events).
/// `vad_model_path` points to the Silero VAD ONNX model file.
pub fn start_capture(
    device_name: Option<&str>,
    event_tx: Sender<AudioEvent>,
    vad_model_path: &Path,
    app_handle: tauri::AppHandle,
) -> Result<CaptureHandle, String> {
    let host = cpal::default_host();

    // Select device
    let device = if let Some(name) = device_name {
        host.input_devices()
            .map_err(|e| format!("enumerate devices: {e}"))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| format!("device not found: {name}"))?
    } else {
        host.default_input_device()
            .ok_or("no default input device")?
    };

    let device_name_str = device.name().unwrap_or("unknown".into());
    log::info!("[audio] using device: {device_name_str}");

    // Get supported config — prefer f32, mono
    let supported = device
        .default_input_config()
        .map_err(|e| format!("no supported input config: {e}"))?;

    let native_rate = supported.sample_rate().0;
    let native_channels = supported.channels() as usize;
    let sample_format = supported.sample_format();

    log::info!(
        "[audio] native: {}Hz, {}ch, {:?}",
        native_rate,
        native_channels,
        sample_format
    );

    let config = StreamConfig {
        channels: supported.channels(),
        sample_rate: SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Channel from cpal callback → processing thread
    // Bounded(32) — ~680ms at 48kHz mono with typical 1024-sample callbacks
    let (raw_tx, raw_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(32);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // Build input stream based on sample format
    let stream = match sample_format {
        SampleFormat::F32 => {
            let channels = native_channels;
            device
                .build_input_stream(
                    &config,
                    move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                        // Downmix to mono if stereo
                        let mono = if channels > 1 {
                            downmix_to_mono(data, channels)
                        } else {
                            data.to_vec()
                        };
                        let _ = raw_tx.try_send(mono); // bounded — drops if full
                    },
                    move |err| log::error!("[audio] stream error: {err}"),
                    None,
                )
                .map_err(|e| format!("build input stream: {e}"))?
        }
        SampleFormat::I16 => {
            let channels = native_channels;
            device
                .build_input_stream(
                    &config,
                    move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                        let f32_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        let mono = if channels > 1 {
                            downmix_to_mono(&f32_data, channels)
                        } else {
                            f32_data
                        };
                        let _ = raw_tx.try_send(mono);
                    },
                    move |err| log::error!("[audio] stream error: {err}"),
                    None,
                )
                .map_err(|e| format!("build input stream (i16): {e}"))?
        }
        fmt => return Err(format!("unsupported sample format: {fmt:?}")),
    };

    stream.play().map_err(|e| format!("stream play: {e}"))?;
    log::info!("[audio] capture started");

    // Spawn processing thread
    let vad_path = vad_model_path.to_path_buf();
    std::thread::Builder::new()
        .name("voxpad-audio-proc".into())
        .spawn(move || {
            if let Err(e) =
                processing_loop(raw_rx, event_tx, &vad_path, native_rate, stop_clone, app_handle)
            {
                log::error!("[audio] processing thread error: {e}");
            }
        })
        .map_err(|e| format!("spawn processing thread: {e}"))?;

    Ok(CaptureHandle {
        _stream: stream,
        stop,
    })
}

/// Main processing loop — resample, VAD, dispatch to ASR.
fn processing_loop(
    raw_rx: Receiver<Vec<f32>>,
    event_tx: Sender<AudioEvent>,
    vad_model_path: &Path,
    native_rate: u32,
    stop: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // Initialize resampler if needed
    let needs_resample = native_rate != TARGET_SAMPLE_RATE;
    let mut resampler = if needs_resample {
        let ratio = TARGET_SAMPLE_RATE as f64 / native_rate as f64;
        log::info!(
            "[audio] resampling {}Hz → {}Hz (ratio {:.4})",
            native_rate,
            TARGET_SAMPLE_RATE,
            ratio
        );
        Some(SimpleResampler::new(ratio))
    } else {
        None
    };

    // Initialize VAD
    let mut vad = SileroVad::new(vad_model_path, 0.5)
        .map_err(|e| format!("VAD init: {e}"))?;

    // Buffers
    let mut resampled_buf: Vec<f32> = Vec::with_capacity(TARGET_SAMPLE_RATE as usize); // 1s
    let mut utterance_audio: Vec<f32> = Vec::with_capacity(TARGET_SAMPLE_RATE as usize * 30); // 30s max
    let mut nemotron_chunk_buf: Vec<f32> = Vec::with_capacity(NEMOTRON_CHUNK_SAMPLES);
    let max_utterance_samples = TARGET_SAMPLE_RATE as usize * 60; // 60s cap

    while let Ok(raw_chunk) = raw_rx.recv() {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Resample to 16kHz
        let resampled = if let Some(ref mut rs) = resampler {
            rs.process(&raw_chunk)
        } else {
            raw_chunk
        };
        resampled_buf.extend_from_slice(&resampled);

        // Process in 512-sample VAD frames (32ms at 16kHz)
        while resampled_buf.len() >= VAD_FRAME_SAMPLES {
            let frame: Vec<f32> = resampled_buf.drain(..VAD_FRAME_SAMPLES).collect();

            match vad.detect(&frame) {
                Ok(event) => {
                    match event {
                        VadEvent::SpeechStart => {
                            log::debug!("[audio] speech start");
                            app_handle.emit("vad-speech-start", ()).ok();
                            let _ = event_tx.try_send(AudioEvent::VadEvent(VadEvent::SpeechStart));
                            // Include this frame in the utterance
                            utterance_audio.extend_from_slice(&frame);
                            nemotron_chunk_buf.extend_from_slice(&frame);
                        }
                        VadEvent::Speaking => {
                            utterance_audio.extend_from_slice(&frame);
                            nemotron_chunk_buf.extend_from_slice(&frame);

                            // Dispatch 560ms chunks to Nemotron
                            while nemotron_chunk_buf.len() >= NEMOTRON_CHUNK_SAMPLES {
                                let chunk: Vec<f32> =
                                    nemotron_chunk_buf.drain(..NEMOTRON_CHUNK_SAMPLES).collect();
                                let _ = event_tx.try_send(AudioEvent::NemotronChunk(chunk));
                            }

                            // Cap utterance at 60 seconds
                            if utterance_audio.len() >= max_utterance_samples {
                                log::info!("[audio] utterance capped at 60s");
                                flush_utterance(
                                    &mut utterance_audio,
                                    &mut nemotron_chunk_buf,
                                    &event_tx,
                                    &app_handle,
                                );
                                vad.reset();
                            }
                        }
                        VadEvent::SpeechEnd => {
                            log::debug!(
                                "[audio] speech end, utterance={:.1}s",
                                utterance_audio.len() as f64 / TARGET_SAMPLE_RATE as f64
                            );
                            flush_utterance(
                                &mut utterance_audio,
                                &mut nemotron_chunk_buf,
                                &event_tx,
                                &app_handle,
                            );
                        }
                        VadEvent::Silence => {
                            // Nothing to do
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[audio] VAD error: {e}");
                }
            }
        }
    }

    log::info!("[audio] processing loop ended");
    Ok(())
}

/// Flush accumulated utterance audio to TDT and any remaining Nemotron chunk.
fn flush_utterance(
    utterance_audio: &mut Vec<f32>,
    nemotron_chunk_buf: &mut Vec<f32>,
    event_tx: &Sender<AudioEvent>,
    app_handle: &tauri::AppHandle,
) {
    // Send remaining partial Nemotron chunk (padded with silence if needed)
    if !nemotron_chunk_buf.is_empty() {
        // Pad to full chunk size for Nemotron
        nemotron_chunk_buf.resize(NEMOTRON_CHUNK_SAMPLES, 0.0);
        let _ = event_tx.try_send(AudioEvent::NemotronChunk(nemotron_chunk_buf.clone()));
        nemotron_chunk_buf.clear();
    }

    // Send complete utterance to TDT
    let min_utterance_samples = (TARGET_SAMPLE_RATE as f64 * 0.2) as usize; // 200ms min
    if utterance_audio.len() >= min_utterance_samples {
        let utterance = std::mem::take(utterance_audio);
        let _ = event_tx.try_send(AudioEvent::TdtUtterance(utterance));
    } else {
        utterance_audio.clear();
    }

    app_handle.emit("vad-speech-end", ()).ok();
    let _ = event_tx.try_send(AudioEvent::VadEvent(VadEvent::SpeechEnd));
}

/// Simple linear interpolation resampler.
/// Good enough for ASR — models are robust to resampling artifacts.
/// Avoids rubato crate API complexity.
struct SimpleResampler {
    ratio: f64,
    fractional_pos: f64,
}

impl SimpleResampler {
    fn new(ratio: f64) -> Self {
        Self {
            ratio,
            fractional_pos: 0.0,
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }

        let output_len = ((input.len() as f64) * self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        while self.fractional_pos < input.len() as f64 - 1.0 {
            let idx = self.fractional_pos as usize;
            let frac = self.fractional_pos - idx as f64;

            // Linear interpolation between adjacent samples
            let sample = if idx + 1 < input.len() {
                input[idx] * (1.0 - frac as f32) + input[idx + 1] * frac as f32
            } else {
                input[idx]
            };
            output.push(sample);

            self.fractional_pos += 1.0 / self.ratio;
        }

        // Carry over fractional position for next chunk
        self.fractional_pos -= input.len() as f64;
        if self.fractional_pos < 0.0 {
            self.fractional_pos = 0.0;
        }

        output
    }
}

/// Downmix multi-channel audio to mono by averaging channels.
fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    let frames = data.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += data[i * channels + ch];
        }
        mono.push(sum / channels as f32);
    }
    mono
}
