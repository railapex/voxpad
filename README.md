# VoxPad

**See what you're saying.** A floating composition buffer with real-time streaming voice transcription.

## What It Does

VoxPad is a system-level tool that sits between your voice and your apps. Hold a key, speak, see your words assemble in real-time — punctuated and cleaned — then release to insert at your cursor. Or tap to open a persistent buffer for longer composition.

- **Quick mode** (hold Insert): Dictate → see text assemble → release → injected at cursor
- **Buffer mode** (tap Insert): Compose over time → Ctrl+Enter to insert when ready
- Real-time streaming ASR with native punctuation and capitalization
- Two-pass: Nemotron 0.6B streaming + Parakeet TDT 0.6B refinement
- Filler word removal (um, uh, etc.)
- Clipboard preserved after injection (including Win+V history exclusion)
- All local — no cloud required for core functionality

## Tech Stack

- **Tauri 2** + Rust backend + vanilla JS frontend
- **NVIDIA Parakeet** ecosystem (Nemotron streaming, TDT batch) via `parakeet-rs`
- **Silero VAD** (direct ort implementation)
- **ONNX Runtime** with CUDA → DirectML → CPU fallback
- **cpal** for audio capture, linear resampling to 16kHz
- **SQLite** with FTS5 for transcription history

## Requirements

- Windows 11 (macOS/Linux planned)
- NVIDIA GPU recommended (CUDA 11.6+), works on CPU
- ~5GB disk for models (downloaded on first run)

## Development

```bash
npm install
cd src-tauri && cargo build
npm run tauri dev
```

## Status

V1 in development. Core pipeline (audio → VAD → ASR → display → inject) is implemented. Model download, history, settings UI complete. Integration and testing in progress.

## License

MIT
