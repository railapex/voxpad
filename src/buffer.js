// VoxPad buffer — streaming text display, mode management, insertion triggers

const textarea = document.getElementById('buffer');
const statusMode = document.getElementById('mode');
const statusWc = document.getElementById('wc');
const recordingDot = document.getElementById('recording-dot');

let mode = 'idle';
let textBuffer = '';
let pendingFlush = null;

function updateWordCount() {
  const text = textarea.value.trim();
  const count = text ? text.split(/\s+/).length : 0;
  statusWc.textContent = `${count} word${count !== 1 ? 's' : ''}`;
}

// Batch streaming text to one DOM update per animation frame
function scheduleFlush() {
  if (pendingFlush) return;
  pendingFlush = requestAnimationFrame(() => {
    if (textBuffer) {
      const wasAtBottom = textarea.scrollHeight - textarea.scrollTop <= textarea.clientHeight + 20;
      textarea.value += textBuffer;
      textBuffer = '';
      if (wasAtBottom) {
        textarea.scrollTop = textarea.scrollHeight;
      }
      updateWordCount();
    }
    pendingFlush = null;
  });
}

// Tauri event listeners will be wired once the Tauri API is available
async function initEvents() {
  try {
    const { listen } = await import('@tauri-apps/api/event');
    const { invoke } = await import('@tauri-apps/api/core');
    const { getCurrentWindow } = await import('@tauri-apps/api/window');
    const appWindow = getCurrentWindow();

    listen('streaming-text', (event) => {
      const { text } = event.payload;
      if (text) {
        textBuffer += text;
        scheduleFlush();
      }
    });

    listen('refined-text', (event) => {
      // V1: TDT refinement is logged to history but not visually replaced.
      // Nemotron streaming output is punctuated and good enough for display.
    });

    listen('buffer-command', (event) => {
      // Backend executed command (scratch/clear) — sync text from backend
      const { text } = event.payload;
      textarea.value = text || '';
      updateWordCount();
    });

    listen('models-loading', () => {
      textarea.value = '';
      textarea.placeholder = 'Models are still loading... try again in a moment.';
    });

    listen('vad-speech-start', () => {
      recordingDot.classList.add('recording');
    });

    listen('vad-speech-end', () => {
      recordingDot.classList.remove('recording');
    });

    listen('enter-quick-mode', async () => {
      mode = 'quick';
      statusMode.textContent = 'QUICK';
      statusMode.className = 'mode-quick';
      textarea.value = '';
      textBuffer = '';
      await appWindow.show();
      await appWindow.setFocus();
    });

    listen('enter-buffer-mode', async () => {
      mode = 'buffer';
      statusMode.textContent = 'BUFFER';
      statusMode.className = 'mode-buffer';
      await appWindow.show();
      await appWindow.setFocus();
    });

    listen('hide-buffer', async () => {
      mode = 'idle';
      statusMode.textContent = 'IDLE';
      statusMode.className = '';
      recordingDot.classList.remove('recording');
      document.body.classList.add('hiding');
      setTimeout(async () => {
        await appWindow.hide();
        document.body.classList.remove('hiding');
      }, 150);
    });

    // Keyboard handlers for buffer mode
    document.addEventListener('keydown', async (e) => {
      if (mode === 'buffer' && e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const text = textarea.value.trim();
        if (text) {
          await invoke('insert_buffer_text', { text });
        }
      }
      if (e.key === 'Escape' && mode === 'buffer') {
        e.preventDefault();
        await invoke('dismiss_buffer');
      }
    });

    console.log('[voxpad] events initialized');
  } catch (err) {
    console.error('[voxpad] event init failed:', err);
  }
}

initEvents();
updateWordCount();

// Also update word count on manual typing
textarea.addEventListener('input', updateWordCount);
