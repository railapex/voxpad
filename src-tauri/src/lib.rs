use tauri::Emitter;
use tauri::Manager;

mod asr;
mod audio;
mod buffer;
mod config;
mod download;
mod history;
mod platform;
mod rules;
mod vad;

use crossbeam_channel::bounded;
use std::sync::{Mutex, OnceLock};

// Global channels for audio→ASR pipeline (separate for streaming vs refinement)
static NEMOTRON_TX: OnceLock<crossbeam_channel::Sender<Vec<f32>>> = OnceLock::new();
static TDT_TX: OnceLock<crossbeam_channel::Sender<Vec<f32>>> = OnceLock::new();
static CAPTURE_HANDLE: OnceLock<Mutex<Option<audio::CaptureHandle>>> = OnceLock::new();

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let data_dir = app.handle().path().app_data_dir().unwrap_or_default();
            let _ = std::fs::create_dir_all(&data_dir);

            // 1. Config
            config::init(&data_dir);
            let cfg = config::get();
            log::info!("[voxpad] started, data_dir={}", data_dir.display());
            log::info!("[voxpad] hotkey={}, hold_threshold={}ms", cfg.hotkey, cfg.hold_threshold_ms);

            // 2. History DB
            if let Err(e) = history::init(&data_dir) {
                log::error!("[voxpad] history init: {e}");
            }

            // 3. Buffer state
            buffer::init();

            // 4. Initialize ORT environment (once, shared by all models)
            ort::init().commit();

            // 5. Create channels — Nemotron disabled until empty-text issue is resolved
            let (nemotron_tx, _nemotron_rx) = bounded(8);
            let (tdt_tx, tdt_rx) = crossbeam_channel::unbounded();
            NEMOTRON_TX.set(nemotron_tx).ok();
            TDT_TX.set(tdt_tx).ok();
            CAPTURE_HANDLE.set(Mutex::new(None)).ok();

            // 6. Spawn TDT thread only — Nemotron disabled (returns empty text, needs investigation)
            // TODO: investigate Nemotron empty output — possibly cache-aware state issue or API mismatch
            // asr::spawn_nemotron_thread(nemotron_rx, app.handle().clone());
            asr::spawn_tdt_thread(tdt_rx, app.handle().clone());

            // 7. System tray
            setup_tray(app)?;

            // 8. Register global hotkey
            register_hotkey(app.handle(), &cfg.hotkey)?;

            // 9. Check models and either preload or show setup wizard
            let models_dir = data_dir.join("models");
            let missing = download::check_models(&data_dir);
            if missing.is_empty() {
                // All models present — preload in background
                log::info!("[voxpad] all models found, preloading...");
                let dir = models_dir.clone();
                std::thread::Builder::new()
                    .name("voxpad-preload".into())
                    .spawn(move || {
                        asr::preload(&dir);
                    })
                    .ok();
            } else {
                // First run — show setup wizard for model download
                log::info!(
                    "[voxpad] {} model files missing, showing setup wizard",
                    missing.len()
                );
                let app_handle = app.handle().clone();
                open_setup_window(&app_handle);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            insert_buffer_text,
            dismiss_buffer,
            get_buffer_text,
            set_buffer_text,
            check_models,
            download_models_cmd,
            cancel_download,
            preload_models,
            get_history,
            search_history,
        ])
        .run(tauri::generate_context!())
        .expect("error running voxpad");
}

// ---------------------------------------------------------------------------
// Hotkey handler
// ---------------------------------------------------------------------------

fn register_hotkey(
    app_handle: &tauri::AppHandle,
    hotkey: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

    let handle = app_handle.clone();
    let hold_threshold = config::get().hold_threshold_ms;

    app_handle.global_shortcut().on_shortcut(hotkey, move |_app, _shortcut, event| {
        match event.state() {
            ShortcutState::Pressed => {
                on_hotkey_pressed(&handle);
            }
            ShortcutState::Released => {
                on_hotkey_released(&handle, hold_threshold);
            }
        }
    })?;

    log::info!("[voxpad] hotkey registered: {}", hotkey);
    Ok(())
}

fn on_hotkey_pressed(app: &tauri::AppHandle) {
    // Don't allow hotkey if models aren't loaded yet
    if !asr::is_ready() {
        log::warn!("[voxpad] models not loaded yet, ignoring hotkey");
        // Show the main window briefly with a "loading" message
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.set_focus();
        }
        app.emit("models-loading", ()).ok();
        return;
    }

    let mode_before = buffer::get_mode();
    let _captured = buffer::on_hotkey_pressed();

    match mode_before {
        buffer::Mode::Idle => {
            // Show window and start recording immediately
            show_buffer_window(app);
            start_recording(app);
            app.emit("enter-quick-mode", ()).ok();
        }
        buffer::Mode::BufferActive => {
            // Toggle off — stop recording and hide
            stop_recording();
            hide_buffer_window(app);
        }
        _ => {}
    }
}

fn on_hotkey_released(app: &tauri::AppHandle, hold_threshold_ms: u64) {
    if !asr::is_ready() {
        return;
    }

    match buffer::on_hotkey_released(hold_threshold_ms) {
        buffer::HotkeyAction::EnterBufferMode => {
            // Was a tap — switch from quick mode visual to buffer mode
            app.emit("enter-buffer-mode", ()).ok();
            // Recording already started on press — continues
        }
        buffer::HotkeyAction::QuickInsert { text, target } => {
            // Was a hold — inject and hide
            stop_recording();
            hide_buffer_window(app);
            if !text.trim().is_empty() {
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    platform::inject_text(&text, target.as_ref());
                });
            }
        }
        buffer::HotkeyAction::HideBuffer => {
            stop_recording();
            hide_buffer_window(app);
        }
        buffer::HotkeyAction::None => {}
    }
}

// ---------------------------------------------------------------------------
// Audio recording control
// ---------------------------------------------------------------------------

fn start_recording(app: &tauri::AppHandle) {
    let nemotron_tx = match NEMOTRON_TX.get() {
        Some(tx) => tx.clone(),
        None => {
            log::error!("[voxpad] nemotron channel not initialized");
            return;
        }
    };
    let tdt_tx = match TDT_TX.get() {
        Some(tx) => tx.clone(),
        None => {
            log::error!("[voxpad] tdt channel not initialized");
            return;
        }
    };

    let data_dir = app.path().app_data_dir().unwrap_or_default();
    let vad_path = data_dir.join("models/silero_vad.onnx");

    if !vad_path.exists() {
        log::error!("[voxpad] VAD model not found at {}", vad_path.display());
        return;
    }

    let mic_device = config::get().mic_device;
    let app_clone = app.clone();

    let channels = audio::AsrChannels {
        nemotron_tx,
        tdt_tx,
    };

    std::thread::spawn(move || {
        match audio::start_capture(
            mic_device.as_deref(),
            channels,
            &vad_path,
            app_clone,
        ) {
            Ok(handle) => {
                if let Some(capture) = CAPTURE_HANDLE.get() {
                    if let Ok(mut lock) = capture.lock() {
                        *lock = Some(handle);
                    }
                }
                log::info!("[voxpad] recording started");
            }
            Err(e) => {
                log::error!("[voxpad] start recording failed: {e}");
            }
        }
    });
}

fn stop_recording() {
    if let Some(capture) = CAPTURE_HANDLE.get() {
        if let Ok(mut lock) = capture.lock() {
            if lock.is_some() {
                *lock = None; // Drop the handle → stops capture
                log::info!("[voxpad] recording stopped");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Window management
// ---------------------------------------------------------------------------

fn show_buffer_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn hide_buffer_window(app: &tauri::AppHandle) {
    app.emit("hide-buffer", ()).ok();
}

fn open_setup_window(app: &tauri::AppHandle) {
    use tauri::WebviewWindowBuilder;
    match WebviewWindowBuilder::new(app, "setup", tauri::WebviewUrl::App("setup.html".into()))
        .title("VoxPad Setup")
        .inner_size(450.0, 350.0)
        .resizable(false)
        .center()
        .build()
    {
        Ok(_) => log::info!("[voxpad] setup window opened"),
        Err(e) => log::error!("[voxpad] failed to open setup window: {e}"),
    }
}

// ---------------------------------------------------------------------------
// System tray
// ---------------------------------------------------------------------------

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{MenuBuilder, MenuItemBuilder};
    use tauri::tray::TrayIconBuilder;

    let settings_item = MenuItemBuilder::with_id("settings", "Settings...").build(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&settings_item, &quit_item])
        .build()?;

    let _tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .tooltip("VoxPad — See what you're saying")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "settings" => {
                log::info!("[tray] settings clicked");
                // TODO: open settings window (same pattern as setup)
            }
            "quit" => {
                log::info!("[tray] quit");
                app.exit(0);
            }
            _ => {}
        })
        .build(app)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn get_config() -> config::Config {
    config::get()
}

/// Insert buffer text at the current foreground window (Ctrl+Enter in buffer mode).
#[tauri::command]
fn insert_buffer_text(text: String, app: tauri::AppHandle) {
    log::info!("[cmd] insert_buffer_text ({} chars)", text.len());
    buffer::set_text(text.clone());
    let final_text = buffer::insert_at_foreground();

    stop_recording();
    hide_buffer_window(&app);

    if !final_text.trim().is_empty() {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            platform::inject_text(&final_text, None);
        });
    }
}

/// Dismiss buffer without inserting (Escape).
#[tauri::command]
fn dismiss_buffer(app: tauri::AppHandle) {
    log::info!("[cmd] dismiss_buffer");
    buffer::dismiss();
    stop_recording();
    hide_buffer_window(&app);
}

#[tauri::command]
fn get_buffer_text() -> String {
    buffer::get_text()
}

#[tauri::command]
fn set_buffer_text(text: String) {
    buffer::set_text(text);
}

/// Check which models need downloading.
#[tauri::command]
fn check_models(app: tauri::AppHandle) -> Vec<download::MissingModel> {
    let data_dir = app.path().app_data_dir().unwrap_or_default();
    download::check_models(&data_dir)
}

/// Download missing models.
#[tauri::command]
async fn download_models_cmd(app: tauri::AppHandle) -> Result<(), String> {
    let data_dir = app.path().app_data_dir().unwrap_or_default();
    download::download_models(data_dir, app.clone()).await
}

/// Cancel an in-progress download.
#[tauri::command]
fn cancel_download() {
    download::cancel();
}

/// Trigger model preload after download completes.
#[tauri::command]
fn preload_models(app: tauri::AppHandle) {
    let data_dir = app.path().app_data_dir().unwrap_or_default();
    let models_dir = data_dir.join("models");
    std::thread::Builder::new()
        .name("voxpad-preload".into())
        .spawn(move || {
            asr::preload(&models_dir);
            log::info!("[voxpad] models preloaded, ready for use");
        })
        .ok();
}

/// Get recent history entries.
#[tauri::command]
fn get_history(limit: Option<u32>, offset: Option<u32>) -> Result<Vec<history::HistoryEntry>, String> {
    history::query_recent(limit.unwrap_or(50), offset.unwrap_or(0))
}

/// Search history via FTS5.
#[tauri::command]
fn search_history(query: String, limit: Option<u32>) -> Result<Vec<history::HistoryEntry>, String> {
    history::search(&query, limit.unwrap_or(50))
}
