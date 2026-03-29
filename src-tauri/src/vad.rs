// Direct Silero VAD implementation via ort.
// Model: onnx-community/silero-vad (silero_vad.onnx)
//
// Inputs:
//   input: float32[1, chunk_size]  — audio samples
//   state: float32[2, 1, 128]     — LSTM hidden state
//   sr:    int64 scalar            — sample rate (16000)
//
// Outputs:
//   output: float32[1, 1]         — speech probability
//   stateN: float32[2, 1, 128]    — updated hidden state

use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const CHUNK_SIZE: usize = 512; // 32ms at 16kHz
const SAMPLE_RATE: i64 = 16000;
const HIDDEN_DIM: usize = 128;
const NUM_LAYERS: usize = 2;
const STATE_SIZE: usize = NUM_LAYERS * 1 * HIDDEN_DIM; // [2, 1, 128] = 256 floats

pub struct SileroVad {
    session: Session,
    /// Combined LSTM state — shape [2, 1, 128]
    state: Vec<f32>,
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

        log::info!("[vad] model loaded from {}", model_path.display());

        Ok(Self {
            session,
            state: vec![0.0f32; STATE_SIZE],
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

        // Input tensor: [1, chunk_size]
        let input = Tensor::from_array((vec![1i64, CHUNK_SIZE as i64], audio.to_vec()))
            .map_err(|e| format!("VAD input tensor: {e}"))?;

        // State tensor: [2, 1, 128]
        let state_tensor = Tensor::from_array((
            vec![NUM_LAYERS as i64, 1i64, HIDDEN_DIM as i64],
            self.state.clone(),
        ))
        .map_err(|e| format!("VAD state tensor: {e}"))?;

        // Sample rate: scalar int64
        let sr = Tensor::from_array((vec![1i64], vec![SAMPLE_RATE]))
            .map_err(|e| format!("VAD sr tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input,
                "state" => state_tensor,
                "sr" => sr,
            ])
            .map_err(|e| format!("VAD inference: {e}"))?;

        // Extract speech probability — output[0] "output"
        let (_, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("VAD output prob: {e}"))?;
        let prob = prob_data[0];

        // Update state from output[1] "stateN"
        let (_, new_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("VAD output state: {e}"))?;
        self.state = new_state.to_vec();

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

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
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
    Silence,
    SpeechStart,
    Speaking,
    SpeechEnd,
}
