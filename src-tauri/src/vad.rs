// Direct Silero VAD implementation via ort.
// No third-party crate needed — the model is a simple LSTM:
//   Input: audio chunk (512 samples at 16kHz), sample rate, hidden state (h, c)
//   Output: speech probability [0.0, 1.0], updated hidden state
//
// Reference: https://github.com/snakers4/silero-vad

use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const CHUNK_SIZE: usize = 512; // 32ms at 16kHz
const SAMPLE_RATE: i64 = 16000;
const HIDDEN_DIM: usize = 64;
const NUM_LAYERS: usize = 2;

pub struct SileroVad {
    session: Session,
    /// Hidden state h — shape [2, 1, 64]
    h: Vec<f32>,
    /// Cell state c — shape [2, 1, 64]
    c: Vec<f32>,
    /// Speech probability threshold
    threshold: f32,
    /// Consecutive silence frames needed to declare end-of-speech
    min_silence_frames: u32,
    /// Consecutive speech frames needed to declare start-of-speech
    min_speech_frames: u32,

    // State tracking
    speech_frames: u32,
    silence_frames: u32,
    is_speaking: bool,
}

impl SileroVad {
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("VAD session builder: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| format!("VAD intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("VAD load model: {e}"))?;

        let state_size = NUM_LAYERS * 1 * HIDDEN_DIM; // [2, 1, 64] = 128 floats
        log::info!("[vad] model loaded from {}", model_path.display());

        Ok(Self {
            session,
            h: vec![0.0f32; state_size],
            c: vec![0.0f32; state_size],
            threshold,
            min_silence_frames: 10, // ~320ms at 32ms/frame
            min_speech_frames: 3,   // ~96ms at 32ms/frame
            speech_frames: 0,
            silence_frames: 0,
            is_speaking: false,
        })
    }

    /// Process a 512-sample audio chunk (32ms at 16kHz).
    /// Returns the raw speech probability [0.0, 1.0].
    pub fn process_chunk(&mut self, audio: &[f32]) -> Result<f32, String> {
        assert_eq!(audio.len(), CHUNK_SIZE, "VAD expects {CHUNK_SIZE} samples per chunk");

        // Create input tensors using ort v2 API: Tensor::from_array((shape, data))
        let input = Tensor::from_array((vec![1i64, CHUNK_SIZE as i64], audio.to_vec()))
            .map_err(|e| format!("VAD input tensor: {e}"))?;
        let sr = Tensor::from_array((vec![1i64], vec![SAMPLE_RATE]))
            .map_err(|e| format!("VAD sr tensor: {e}"))?;
        let h_tensor = Tensor::from_array((
            vec![NUM_LAYERS as i64, 1i64, HIDDEN_DIM as i64],
            self.h.clone(),
        ))
        .map_err(|e| format!("VAD h tensor: {e}"))?;
        let c_tensor = Tensor::from_array((
            vec![NUM_LAYERS as i64, 1i64, HIDDEN_DIM as i64],
            self.c.clone(),
        ))
        .map_err(|e| format!("VAD c tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input,
                "sr" => sr,
                "h" => h_tensor,
                "c" => c_tensor,
            ])
            .map_err(|e| format!("VAD inference: {e}"))?;

        // Extract speech probability — output[0] is shape [1, 1]
        let (_, prob_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("VAD output prob: {e}"))?;
        let prob = prob_data[0];

        // Update hidden states from outputs[1] and outputs[2]
        let (_, new_h) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("VAD output h: {e}"))?;
        let (_, new_c) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("VAD output c: {e}"))?;

        self.h = new_h.to_vec();
        self.c = new_c.to_vec();

        Ok(prob)
    }

    /// Process a chunk and return a speech state transition event.
    pub fn detect(&mut self, audio: &[f32]) -> Result<VadEvent, String> {
        let prob = self.process_chunk(audio)?;

        if prob >= self.threshold {
            self.speech_frames += 1;
            self.silence_frames = 0;

            if !self.is_speaking && self.speech_frames >= self.min_speech_frames {
                self.is_speaking = true;
                return Ok(VadEvent::SpeechStart);
            }
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;

            if self.is_speaking && self.silence_frames >= self.min_silence_frames {
                self.is_speaking = false;
                return Ok(VadEvent::SpeechEnd);
            }
        }

        if self.is_speaking {
            Ok(VadEvent::Speaking)
        } else {
            Ok(VadEvent::Silence)
        }
    }

    /// Reset internal state (between utterances or on mode change).
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.is_speaking = false;
    }

    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadEvent {
    /// Silence — no speech detected
    Silence,
    /// Speech just started (transition from silence → speech)
    SpeechStart,
    /// Ongoing speech
    Speaking,
    /// Speech just ended (transition from speech → silence)
    SpeechEnd,
}
