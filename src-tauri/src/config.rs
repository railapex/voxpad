use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{OnceLock, RwLock};

static CONFIG: OnceLock<RwLock<Config>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_hotkey")]
    pub hotkey: String,
    #[serde(default)]
    pub mic_device: Option<String>,
    #[serde(default)]
    pub buffer_x: Option<i32>,
    #[serde(default)]
    pub buffer_y: Option<i32>,
    #[serde(default)]
    pub buffer_width: Option<u32>,
    #[serde(default)]
    pub buffer_height: Option<u32>,
    #[serde(default)]
    pub start_at_login: bool,
    #[serde(default = "default_hold_threshold_ms")]
    pub hold_threshold_ms: u64,
}

fn default_hotkey() -> String {
    "Insert".to_string()
}

fn default_hold_threshold_ms() -> u64 {
    250
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey(),
            mic_device: None,
            buffer_x: None,
            buffer_y: None,
            buffer_width: None,
            buffer_height: None,
            start_at_login: false,
            hold_threshold_ms: default_hold_threshold_ms(),
        }
    }
}

pub fn init(data_dir: &Path) {
    let config_path = data_dir.join("config.json");
    let config = if config_path.exists() {
        match std::fs::read_to_string(&config_path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Config::default(),
        }
    } else {
        Config::default()
    };
    CONFIG.set(RwLock::new(config)).ok();
}

pub fn get() -> Config {
    CONFIG
        .get()
        .expect("config not initialized")
        .read()
        .unwrap()
        .clone()
}

pub fn update<F: FnOnce(&mut Config)>(data_dir: &Path, f: F) -> Config {
    let lock = CONFIG.get().expect("config not initialized");
    let mut config = lock.write().unwrap();
    f(&mut config);
    let updated = config.clone();
    drop(config);
    save(data_dir, &updated);
    updated
}

fn save(data_dir: &Path, config: &Config) {
    let _ = std::fs::create_dir_all(data_dir);
    let json = serde_json::to_string_pretty(config).unwrap_or_default();
    let _ = std::fs::write(data_dir.join("config.json"), json);
}

pub fn is_first_run(data_dir: &Path) -> bool {
    !data_dir.join("config.json").exists()
}
