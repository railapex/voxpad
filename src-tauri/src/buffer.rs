// Buffer state machine — Quick mode (hold) and Buffer mode (tap).
//
// HotkeyPressed:
//   → Capture foreground HWND
//   → Show buffer window
//   → Start recording immediately
//   → Start hold_threshold timer
//
// HotkeyReleased < threshold (TAP):
//   → Switch to Buffer mode (recording continues)
//
// HotkeyReleased >= threshold (HOLD):
//   → Quick mode: stop recording, inject text, hide window

use crate::platform::WindowHandle;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static STATE: OnceLock<Mutex<BufferState>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// No active recording, window hidden
    Idle,
    /// Hotkey pressed, recording started, waiting for hold/tap determination
    Pressed,
    /// Buffer mode: window visible, recording, persists after key release
    BufferActive,
}

pub struct BufferState {
    pub mode: Mode,
    /// Accumulated buffer text from ASR
    pub text: String,
    /// Target window for quick mode injection (captured on press)
    pub captured_hwnd: Option<WindowHandle>,
    /// When the hotkey was pressed (for hold/tap detection)
    pub press_time: Option<Instant>,
    /// Track utterance boundaries for "scratch that"
    pub utterance_boundaries: Vec<usize>, // byte offsets where each utterance starts
}

impl BufferState {
    fn new() -> Self {
        Self {
            mode: Mode::Idle,
            text: String::new(),
            captured_hwnd: None,
            press_time: None,
            utterance_boundaries: Vec::new(),
        }
    }
}

/// Initialize the buffer state. Call once at startup.
pub fn init() {
    STATE.set(Mutex::new(BufferState::new())).ok();
}

/// Get the current mode.
pub fn get_mode() -> Mode {
    STATE
        .get()
        .map(|s| s.lock().unwrap().mode)
        .unwrap_or(Mode::Idle)
}

/// Record a hotkey press event. Returns the captured HWND.
pub fn on_hotkey_pressed() -> Option<WindowHandle> {
    let state = STATE.get()?;
    let mut lock = state.lock().ok()?;

    match lock.mode {
        Mode::Idle => {
            lock.mode = Mode::Pressed;
            lock.press_time = Some(Instant::now());

            // Capture foreground before we show our window
            lock.captured_hwnd = crate::platform::capture_foreground();

            log::debug!("[buffer] pressed, captured hwnd: {:?}", lock.captured_hwnd);
            lock.captured_hwnd.clone()
        }
        Mode::BufferActive => {
            // Already in buffer mode — this press will toggle off on release
            lock.mode = Mode::Idle;
            lock.press_time = None;
            log::debug!("[buffer] buffer mode → idle (tap toggle off)");
            None
        }
        _ => None,
    }
}

/// Record a hotkey release event.
/// Returns an action to take based on hold/tap detection.
pub fn on_hotkey_released(hold_threshold_ms: u64) -> HotkeyAction {
    let state = match STATE.get() {
        Some(s) => s,
        None => return HotkeyAction::None,
    };
    let mut lock = match state.lock() {
        Ok(l) => l,
        Err(_) => return HotkeyAction::None,
    };

    match lock.mode {
        Mode::Pressed => {
            let held_ms = lock
                .press_time
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(0);

            if held_ms < hold_threshold_ms {
                // TAP — switch to buffer mode (recording continues)
                lock.mode = Mode::BufferActive;
                lock.press_time = None;
                log::debug!("[buffer] tap detected ({}ms) → buffer mode", held_ms);
                HotkeyAction::EnterBufferMode
            } else {
                // HOLD — quick mode: inject and hide
                let text = lock.text.clone();
                let hwnd = lock.captured_hwnd.clone();
                lock.mode = Mode::Idle;
                lock.text.clear();
                lock.utterance_boundaries.clear();
                lock.captured_hwnd = None;
                lock.press_time = None;
                log::debug!(
                    "[buffer] hold detected ({}ms) → quick inject ({} chars)",
                    held_ms,
                    text.len()
                );
                HotkeyAction::QuickInsert { text, target: hwnd }
            }
        }
        Mode::Idle => {
            // Was BufferActive, toggled off on press — stop recording, hide
            HotkeyAction::HideBuffer
        }
        _ => HotkeyAction::None,
    }
}

/// Append streaming text from ASR to the buffer.
pub fn append_text(text: &str) {
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            if !lock.text.is_empty() && !lock.text.ends_with('\n') {
                lock.text.push(' ');
            }
            let boundary = lock.text.len();
            lock.utterance_boundaries.push(boundary);
            lock.text.push_str(text);
        }
    }
}

/// Get the current buffer text.
pub fn get_text() -> String {
    STATE
        .get()
        .and_then(|s| s.lock().ok())
        .map(|l| l.text.clone())
        .unwrap_or_default()
}

/// Set buffer text (e.g., after user edits in textarea).
pub fn set_text(text: String) {
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            lock.text = text;
            lock.utterance_boundaries.clear(); // can't track boundaries after manual edit
        }
    }
}

/// Handle "scratch that" — remove the last utterance.
/// Returns true if something was removed.
pub fn scratch_last() -> bool {
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            if let Some(boundary) = lock.utterance_boundaries.pop() {
                if boundary <= lock.text.len() {
                    lock.text.truncate(boundary);
                    // Trim trailing whitespace
                    let trimmed_len = lock.text.trim_end().len();
                    lock.text.truncate(trimmed_len);
                    return true;
                }
            }
        }
    }
    false
}

/// Clear the entire buffer.
pub fn clear() {
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            lock.text.clear();
            lock.utterance_boundaries.clear();
        }
    }
}

/// Insert buffer text at the current foreground window (buffer mode Ctrl+Enter).
pub fn insert_at_foreground() -> String {
    let text = get_text();
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            lock.mode = Mode::Idle;
            lock.text.clear();
            lock.utterance_boundaries.clear();
            lock.captured_hwnd = None;
        }
    }
    text
}

/// Dismiss buffer without inserting (Escape).
pub fn dismiss() {
    if let Some(state) = STATE.get() {
        if let Ok(mut lock) = state.lock() {
            lock.mode = Mode::Idle;
            // Keep text in memory for history, but clear state
            lock.captured_hwnd = None;
            lock.press_time = None;
        }
    }
}

#[derive(Debug)]
pub enum HotkeyAction {
    /// No action needed
    None,
    /// Enter buffer mode (recording continues, window stays)
    EnterBufferMode,
    /// Quick mode: inject text at target and hide
    QuickInsert {
        text: String,
        target: Option<WindowHandle>,
    },
    /// Hide the buffer (toggle off from buffer mode)
    HideBuffer,
}
