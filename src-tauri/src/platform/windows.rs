// Windows platform implementation — clipboard injection, window management, DWM recovery.
// Adapted from LinguaLens D:/dev/lingualens/src-tauri/src/lib.rs

use super::WindowHandle;

pub fn capture_foreground() -> Option<WindowHandle> {
    #[cfg(target_os = "windows")]
    unsafe {
        let hwnd = windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow();
        Some(WindowHandle(hwnd.0 as isize))
    }
}

pub fn focus_window(handle: &WindowHandle) {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
        use windows::Win32::Foundation::HWND;
        let _ = SetForegroundWindow(HWND(handle.0 as *mut _));
    }
}

pub fn inject_text(text: &str, _target: Option<&WindowHandle>) {
    // TODO Phase 4: Full clipboard save/restore/inject from LinguaLens
    log::info!("[platform/windows] inject_text stub: {} chars", text.len());
}

// TODO Phase 4: DWM sleep/wake recovery
// pub fn did_system_sleep() -> bool { ... }
// pub fn recreate_overlay_window(app: &tauri::AppHandle) { ... }
