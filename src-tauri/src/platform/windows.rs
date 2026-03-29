// Windows platform implementation — clipboard injection, window management, DWM recovery.
// Adapted from LinguaLens D:/dev/lingualens/src-tauri/src/lib.rs

use super::WindowHandle;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Saved clipboard data
// ---------------------------------------------------------------------------

pub struct SavedClipboardFormat {
    pub format: u32,
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Window capture / focus
// ---------------------------------------------------------------------------

pub fn capture_foreground() -> Option<WindowHandle> {
    unsafe {
        let hwnd = windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow();
        if hwnd.0.is_null() {
            None
        } else {
            Some(WindowHandle(hwnd.0 as isize))
        }
    }
}

pub fn focus_window(handle: &WindowHandle) {
    unsafe {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
        let _ = SetForegroundWindow(HWND(handle.0 as *mut _));
    }
}

// ---------------------------------------------------------------------------
// Clipboard injection — full pipeline
// ---------------------------------------------------------------------------

/// Inject text into the target window via clipboard.
/// 1. Save all clipboard formats
/// 2. Set clipboard to text + exclusion markers
/// 3. Focus target window
/// 4. Simulate Ctrl+V
/// 5. Restore original clipboard after delay
pub fn inject_text(text: &str, target: Option<&WindowHandle>) {
    let saved = save_clipboard_all();
    set_clipboard_text(text);

    if let Some(handle) = target {
        focus_window(handle);
    }

    std::thread::sleep(Duration::from_millis(30));
    simulate_ctrl_v();

    // Restore clipboard after a delay (give paste time to complete)
    let saved = saved;
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let _ = restore_clipboard_all_excluded(&saved);
    });
}

// ---------------------------------------------------------------------------
// Clipboard save/restore (from LinguaLens)
// ---------------------------------------------------------------------------

fn save_clipboard_all() -> Vec<SavedClipboardFormat> {
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::System::DataExchange::*;
    use windows::Win32::System::Memory::*;

    let mut formats = Vec::new();

    unsafe {
        if OpenClipboard(None).is_err() {
            return formats;
        }

        let mut fmt = EnumClipboardFormats(0);
        while fmt != 0 {
            // Skip GDI handle formats — can't save/restore raw bytes
            // CF_BITMAP=2, CF_METAFILEPICT=3, CF_PALETTE=9, CF_ENHMETAFILE=14
            if !matches!(fmt, 2 | 3 | 9 | 14) {
                if let Ok(handle) = GetClipboardData(fmt) {
                    let hmem = HGLOBAL(handle.0);
                    let size = GlobalSize(hmem);
                    if size > 0 {
                        let ptr = GlobalLock(hmem) as *const u8;
                        if !ptr.is_null() {
                            let data = std::slice::from_raw_parts(ptr, size).to_vec();
                            formats.push(SavedClipboardFormat { format: fmt, data });
                            let _ = GlobalUnlock(hmem);
                        }
                    }
                }
            }
            fmt = EnumClipboardFormats(fmt);
        }

        let _ = CloseClipboard();
    }

    formats
}

fn set_clipboard_text(text: &str) {
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::System::DataExchange::*;
    use windows::Win32::System::Memory::*;
    use windows::Win32::System::Ole::CF_UNICODETEXT;

    unsafe {
        if OpenClipboard(None).is_err() {
            log::error!("[clipboard] OpenClipboard failed");
            return;
        }

        let _ = EmptyClipboard();

        let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
        let bytes = wide.len() * 2;

        if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, bytes) {
            let ptr = GlobalLock(HGLOBAL(hmem.0)) as *mut u16;
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr, wide.len());
                let _ = GlobalUnlock(HGLOBAL(hmem.0));
                let _ = SetClipboardData(
                    CF_UNICODETEXT.0 as u32,
                    Some(windows::Win32::Foundation::HANDLE(hmem.0)),
                );
            }
        }

        // Set exclusion markers to prevent Win+V pollution
        set_clipboard_exclusion_formats();

        let _ = CloseClipboard();
    }
}

fn restore_clipboard_all_excluded(formats: &[SavedClipboardFormat]) -> Result<(), String> {
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::System::DataExchange::*;
    use windows::Win32::System::Memory::*;

    unsafe {
        OpenClipboard(None).map_err(|e| format!("OpenClipboard: {e}"))?;

        let result = (|| -> Result<(), String> {
            EmptyClipboard().map_err(|e| format!("EmptyClipboard: {e}"))?;

            for saved in formats {
                if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, saved.data.len()) {
                    let ptr = GlobalLock(HGLOBAL(hmem.0)) as *mut u8;
                    if !ptr.is_null() {
                        std::ptr::copy_nonoverlapping(
                            saved.data.as_ptr(),
                            ptr,
                            saved.data.len(),
                        );
                        let _ = GlobalUnlock(HGLOBAL(hmem.0));
                        let _ = SetClipboardData(
                            saved.format,
                            Some(windows::Win32::Foundation::HANDLE(hmem.0)),
                        );
                    }
                }
            }

            set_clipboard_exclusion_formats();
            Ok(())
        })();

        let _ = CloseClipboard();
        result
    }
}

/// Set clipboard formats that tell clipboard managers to ignore this write.
/// Must be called while clipboard is open.
unsafe fn set_clipboard_exclusion_formats() {
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::System::DataExchange::*;
    use windows::Win32::System::Memory::*;
    use windows::core::w;

    // Win10+ clipboard history (Win+V)
    let fmt1 = RegisterClipboardFormatW(w!("ExcludeClipboardContentFromMonitorProcessing"));
    if fmt1 != 0 {
        if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, 1) {
            let _ = SetClipboardData(fmt1, Some(windows::Win32::Foundation::HANDLE(hmem.0)));
        }
    }

    // CanIncludeInClipboardHistory (DWORD = 0)
    let fmt2 = RegisterClipboardFormatW(w!("CanIncludeInClipboardHistory"));
    if fmt2 != 0 {
        if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, 4) {
            let ptr = GlobalLock(HGLOBAL(hmem.0)) as *mut u32;
            if !ptr.is_null() {
                *ptr = 0;
                let _ = GlobalUnlock(HGLOBAL(hmem.0));
            }
            let _ = SetClipboardData(fmt2, Some(windows::Win32::Foundation::HANDLE(hmem.0)));
        }
    }

    // Ditto and other third-party clipboard managers
    let fmt3 = RegisterClipboardFormatW(w!("Clipboard Viewer Ignore"));
    if fmt3 != 0 {
        if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, 1) {
            let _ = SetClipboardData(fmt3, Some(windows::Win32::Foundation::HANDLE(hmem.0)));
        }
    }
}

// ---------------------------------------------------------------------------
// Simulate Ctrl+V (adapted from LinguaLens simulate_ctrl_c)
// ---------------------------------------------------------------------------

fn simulate_ctrl_v() {
    use windows::Win32::UI::Input::KeyboardAndMouse::*;

    // Release all modifier keys first (user may still be holding hotkey)
    let release_mods = [
        make_key_input(0x11, true), // VK_CONTROL up
        make_key_input(0x12, true), // VK_ALT up
        make_key_input(0x10, true), // VK_SHIFT up
        make_key_input(0x5B, true), // VK_LWIN up
    ];

    unsafe {
        SendInput(&release_mods, std::mem::size_of::<INPUT>() as i32);
    }

    std::thread::sleep(Duration::from_millis(30));

    // Ctrl+V down, V up, Ctrl up
    let paste = [
        make_key_input(0x11, false), // VK_CONTROL down
        make_key_input(0x56, false), // VK_V down
        make_key_input(0x56, true),  // VK_V up
        make_key_input(0x11, true),  // VK_CONTROL up
    ];

    unsafe {
        SendInput(&paste, std::mem::size_of::<INPUT>() as i32);
    }
}

fn make_key_input(vk: u16, key_up: bool) -> windows::Win32::UI::Input::KeyboardAndMouse::INPUT {
    use windows::Win32::UI::Input::KeyboardAndMouse::*;

    let flags = if key_up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };

    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                dwFlags: flags,
                ..Default::default()
            },
        },
    }
}

// ---------------------------------------------------------------------------
// DWM sleep/wake recovery
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};

static LAST_TICK: AtomicU64 = AtomicU64::new(0);

/// Detect system sleep/wake by looking for large jumps in GetTickCount64.
/// Returns true if the system likely slept since the last call.
pub fn did_system_sleep() -> bool {
    let now = unsafe { windows::Win32::System::SystemInformation::GetTickCount64() };
    let prev = LAST_TICK.swap(now, Ordering::Relaxed);
    if prev == 0 {
        return false; // first call — baseline
    }
    // If >5 seconds elapsed between checks that should be <2 seconds apart,
    // the system likely slept. Rough but functional.
    now.saturating_sub(prev) > 5000
}
