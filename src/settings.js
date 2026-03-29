// VoxPad settings — hotkey capture, mic device selection, preferences

async function init() {
  try {
    const { invoke } = await import('@tauri-apps/api/core');

    // Load current config
    const config = await invoke('get_config');

    // Hotkey display
    const hotkeyBtn = document.getElementById('hotkey-btn');
    hotkeyBtn.textContent = config.hotkey || 'Insert';

    // Hold threshold slider
    const thresholdSlider = document.getElementById('hold-threshold');
    const thresholdValue = document.getElementById('hold-threshold-value');
    thresholdSlider.value = config.hold_threshold_ms || 250;
    thresholdValue.textContent = `${thresholdSlider.value}ms`;

    thresholdSlider.addEventListener('input', () => {
      thresholdValue.textContent = `${thresholdSlider.value}ms`;
    });

    // Mic device dropdown
    const micSelect = document.getElementById('mic-device');
    // TODO: invoke('enumerate_mic_devices') and populate dropdown

    // Start at login
    const startToggle = document.getElementById('start-at-login');
    startToggle.checked = config.start_at_login || false;

    // Hotkey capture
    let capturing = false;
    hotkeyBtn.addEventListener('click', () => {
      if (capturing) return;
      capturing = true;
      hotkeyBtn.classList.add('capturing');
      hotkeyBtn.textContent = 'Press a key...';

      const handler = (e) => {
        e.preventDefault();
        const key = e.key;
        if (key === 'Escape') {
          // Cancel capture
          hotkeyBtn.textContent = config.hotkey || 'Insert';
        } else {
          let combo = '';
          if (e.ctrlKey) combo += 'Ctrl+';
          if (e.shiftKey) combo += 'Shift+';
          if (e.altKey) combo += 'Alt+';
          if (!['Control', 'Shift', 'Alt', 'Meta'].includes(key)) {
            combo += key;
            hotkeyBtn.textContent = combo;
            // TODO: invoke('update_hotkey', { hotkey: combo })
          }
        }
        capturing = false;
        hotkeyBtn.classList.remove('capturing');
        document.removeEventListener('keydown', handler);
      };

      document.addEventListener('keydown', handler);
    });

    console.log('[settings] initialized');
  } catch (err) {
    console.error('[settings] init failed:', err);
  }
}

init();
