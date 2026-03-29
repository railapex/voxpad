#[cfg(target_os = "windows")]
pub mod windows;

// #[cfg(target_os = "macos")]
// pub mod macos;

// #[cfg(target_os = "linux")]
// pub mod linux;

/// Opaque handle to a platform window.
#[derive(Debug, Clone)]
pub struct WindowHandle(pub isize);

/// Capture the current foreground window handle.
pub fn capture_foreground() -> Option<WindowHandle> {
    #[cfg(target_os = "windows")]
    return windows::capture_foreground();

    #[cfg(not(target_os = "windows"))]
    {
        log::warn!("[platform] capture_foreground not implemented on this platform");
        None
    }
}

/// Focus a previously captured window.
pub fn focus_window(handle: &WindowHandle) {
    #[cfg(target_os = "windows")]
    windows::focus_window(handle);

    #[cfg(not(target_os = "windows"))]
    log::warn!("[platform] focus_window not implemented on this platform");
}

/// Inject text into the target window via clipboard.
/// Saves clipboard, sets text, simulates Ctrl+V, restores clipboard.
pub fn inject_text(text: &str, target: Option<&WindowHandle>) {
    #[cfg(target_os = "windows")]
    windows::inject_text(text, target);

    #[cfg(not(target_os = "windows"))]
    log::warn!("[platform] inject_text not implemented on this platform");
}
