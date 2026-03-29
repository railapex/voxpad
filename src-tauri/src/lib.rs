use tauri::Manager;
mod config;
mod vad;
mod audio;
mod asr;
mod rules;
// mod buffer;  // Phase 4
// mod history; // Phase 5
// mod download; // Phase 5
mod platform;

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
            config::init(&data_dir);
            log::info!("[voxpad] started, data_dir={}", data_dir.display());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
        ])
        .run(tauri::generate_context!())
        .expect("error running voxpad");
}

#[tauri::command]
fn get_config() -> config::Config {
    config::get()
}
