// VoxPad buffer — streaming text display, mode management, insertion triggers
// Uses window.__TAURI__ globals (withGlobalTauri: true)

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { getCurrentWindow } = window.__TAURI__.window;

const textarea = document.getElementById('buffer');
const statusMode = document.getElementById('mode');
const statusWc = document.getElementById('wc');
const recordingDot = document.getElementById('recording-dot');
const appWindow = getCurrentWindow();

let mode = 'idle';
let textBuffer = '';
let pendingFlush = null;

function updateWordCount() {
  const text = textarea.value.trim();
  const count = text ? text.split(/\s+/).length : 0;
  statusWc.textContent = count + ' word' + (count !== 1 ? 's' : '');
}

// Batch streaming text to one DOM update per animation frame
function scheduleFlush() {
  if (pendingFlush) return;
  pendingFlush = requestAnimationFrame(function() {
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

// Tauri event listeners
listen('streaming-text', function(event) {
  var text = event.payload.text;
  if (text) {
    textBuffer += text + ' ';
    scheduleFlush();
  }
});

listen('refined-text', function(event) {
  // TDT produces the authoritative text — show it
  var text = event.payload.text;
  if (text) {
    textBuffer += text + ' ';
    scheduleFlush();
  }
});

listen('buffer-command', function(event) {
  // Backend executed command (scratch/clear) — sync text
  textarea.value = event.payload.text || '';
  updateWordCount();
});

listen('models-loading', function() {
  textarea.value = '';
  textarea.placeholder = 'Models are still loading... try again in a moment.';
});

listen('vad-speech-start', function() {
  recordingDot.classList.add('recording');
});

listen('vad-speech-end', function() {
  recordingDot.classList.remove('recording');
});

listen('enter-quick-mode', async function() {
  mode = 'quick';
  statusMode.textContent = 'QUICK';
  statusMode.className = 'mode-quick';
  textarea.value = '';
  textBuffer = '';
  textarea.placeholder = 'Speak...';
  await appWindow.show();
  await appWindow.setFocus();
});

listen('enter-buffer-mode', async function() {
  mode = 'buffer';
  statusMode.textContent = 'BUFFER';
  statusMode.className = 'mode-buffer';
  textarea.placeholder = 'Speak or type... Ctrl+Enter to insert, Escape to dismiss.';
  await appWindow.show();
  await appWindow.setFocus();
});

listen('hide-buffer', async function() {
  mode = 'idle';
  statusMode.textContent = 'IDLE';
  statusMode.className = '';
  recordingDot.classList.remove('recording');
  document.body.classList.add('hiding');
  setTimeout(async function() {
    await appWindow.hide();
    document.body.classList.remove('hiding');
  }, 150);
});

// Keyboard handlers
document.addEventListener('keydown', async function(e) {
  if (mode === 'buffer' && e.ctrlKey && e.key === 'Enter') {
    e.preventDefault();
    var text = textarea.value.trim();
    if (text) {
      await invoke('insert_buffer_text', { text: text });
    }
  }
  if (e.key === 'Escape' && mode === 'buffer') {
    e.preventDefault();
    await invoke('dismiss_buffer');
  }
});

// Update word count on manual typing
textarea.addEventListener('input', updateWordCount);
updateWordCount();

console.log('[voxpad] buffer.js initialized');
