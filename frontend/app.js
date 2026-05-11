(() => {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  let currentLang = 'nl';
  // History lives on the server. We mirror it locally only for rendering.
  let history = [];           // [{role, content}, ...]
  let isBusy = false;

  // VAD-driven mic state
  const VAD = {
    OPEN_THRESHOLD:    0.025,  // RMS above this opens the gate
    CLOSE_THRESHOLD:   0.012,  // RMS below this closes the gate (hysteresis)
    OPEN_SUSTAIN_MS:   150,    // must stay above for this long to start recording
    CLOSE_SUSTAIN_MS:  700,    // must stay below for this long to stop recording
    MIN_UTTERANCE_MS:  400,    // drop blips shorter than this
    AUTO_SEND:         true,   // stage flow: send transcribed text immediately
  };

  let micArmed = false;
  let micStream = null;
  let audioCtx = null;
  let analyser = null;
  let rafHandle = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let recState = 'idle';       // 'idle' | 'recording'
  let aboveSinceMs = 0;
  let belowSinceMs = 0;
  let recordStartMs = 0;

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const chat       = document.getElementById('chat');
  const textInput  = document.getElementById('text-input');
  const sendBtn    = document.getElementById('send-btn');
  const micBtn     = document.getElementById('mic-btn');
  const langToggle = document.getElementById('lang-toggle');
  const statusBar  = document.getElementById('status-bar');
  const statusDot  = document.getElementById('status-dot');
  const rmsMeter   = document.getElementById('rms-meter');
  const rmsFill    = document.getElementById('rms-fill');
  const rmsValue   = document.getElementById('rms-value');
  const rmsMarkOpen  = document.getElementById('rms-mark-open');
  const rmsMarkClose = document.getElementById('rms-mark-close');

  // Display ceiling for the meter — RMS rarely exceeds ~0.3 in normal speech.
  const RMS_DISPLAY_MAX = 0.3;

  // ── Helpers ───────────────────────────────────────────────────────────────
  function setStatus(msg, busy = false) {
    statusBar.textContent = msg;
    statusDot.className = busy ? 'busy' : '';
    isBusy = busy;
    sendBtn.disabled = busy;
  }

  function clearStatus() { setStatus('', false); }

  function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function pruneHistory() {
    // Display-only cap; the server is the source of truth.
    if (history.length > 200) history = history.slice(history.length - 200);
  }

  async function loadHistoryFromServer() {
    try {
      const res = await fetch('/history');
      const data = await res.json();
      const msgs = (data && data.history) || [];
      history = msgs.slice();
      for (const m of msgs) {
        appendMessage(m.role === 'assistant' ? 'bot' : 'user', m.content);
      }
    } catch (_) {}
  }

  // ── Language toggle ───────────────────────────────────────────────────────
  async function syncLang() {
    try {
      const res = await fetch('/language');
      const data = await res.json();
      currentLang = data.lang;
      langToggle.textContent = currentLang.toUpperCase();
    } catch (_) {}
  }

  langToggle.addEventListener('click', async () => {
    const next = currentLang === 'nl' ? 'en' : 'nl';
    try {
      await fetch('/language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: next }),
      });
      currentLang = next;
      langToggle.textContent = next.toUpperCase();
    } catch (e) {
      setStatus('Language switch failed.');
    }
  });

  // ── TTS ───────────────────────────────────────────────────────────────────
  function stripBracketedTranslations(text) {
    // Remove parenthetical translations the model sometimes adds, e.g. "(Hello! How are you?)"
    return text.replace(/\s*\([^)]{10,}\)/g, '').trim();
  }

  function cleanForTTS(text) {
    return text
      .replace(/\*\*(.+?)\*\*/gs, '$1')
      .replace(/\*(.+?)\*/gs, '$1')
      .replace(/_(.+?)_/gs, '$1')
      .replace(/`{1,3}[\s\S]*?`{1,3}/g, '')
      .replace(/#{1,6}\s*/g, '')
      .replace(/^[-*]\s+/gm, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Streaming TTS: serial queue. While chunk N plays, chunk N+1 is being synthesized
  // (next iteration's fetch starts as soon as the previous play resolves).
  // Serial — not parallel — because the server's TTS engine is not thread-safe.
  const TTS_MIN_CHUNK_CHARS = 12;
  const SENTENCE_END_RE = /[.!?…]/;

  function extractFlushableSentence(buf) {
    // Returns the longest prefix that ends at a sentence terminator outside any open paren,
    // respecting a minimum length so we don't synthesize one-word fragments.
    let parens = 0;
    let lastEnd = -1;
    for (let i = 0; i < buf.length; i++) {
      const c = buf[i];
      if (c === '(') parens++;
      else if (c === ')') parens = Math.max(0, parens - 1);
      else if (parens === 0 && (SENTENCE_END_RE.test(c) || c === '\n')) {
        lastEnd = i;
      }
    }
    if (lastEnd < TTS_MIN_CHUNK_CHARS - 1) return null;
    return buf.slice(0, lastEnd + 1);
  }

  class TTSPlayer {
    constructor(lang, onFirstPlay) {
      this.lang = lang;
      this.q = [];
      this.busy = false;
      this.idleCbs = [];
      this.cancelled = false;
      this.firstPlayed = false;
      this.onFirstPlay = onFirstPlay;
      this.currentAudio = null;
    }
    enqueue(text) {
      if (this.cancelled) return;
      const t = (text || '').trim();
      if (!t) return;
      this.q.push(t);
      if (!this.busy) this._loop();
    }
    onIdle(cb) {
      if (!this.busy && this.q.length === 0) cb();
      else this.idleCbs.push(cb);
    }
    cancel() {
      this.cancelled = true;
      this.q = [];
      if (this.currentAudio) {
        try { this.currentAudio.pause(); } catch (_) {}
      }
    }
    async _loop() {
      this.busy = true;
      while (this.q.length && !this.cancelled) {
        const text = this.q.shift();
        try {
          const res = await fetch('/synthesize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang: this.lang }),
          });
          if (!res.ok) continue;
          const blob = await res.blob();
          if (this.cancelled) break;
          await this._play(blob);
        } catch (e) {
          console.error('TTS chunk error:', e);
        }
      }
      this.busy = false;
      if (!this.cancelled) {
        while (this.idleCbs.length) (this.idleCbs.shift())();
      }
    }
    _play(blob) {
      return new Promise((resolve) => {
        const url = URL.createObjectURL(blob);
        const a = new Audio(url);
        this.currentAudio = a;
        const done = () => { URL.revokeObjectURL(url); this.currentAudio = null; resolve(); };
        a.onended = done;
        a.onerror = done;
        if (!this.firstPlayed) {
          this.firstPlayed = true;
          if (this.onFirstPlay) {
            try { this.onFirstPlay(); } catch (_) {}
          }
        }
        a.play().catch(done);
      });
    }
  }

  let activeTTS = null;

  // ── LLM chat (WebSocket streaming) ───────────────────────────────────────
  function sendMessage(userText) {
    if (!userText.trim() || isBusy) return;

    // If a previous TTS was still going (e.g. user interrupted), tear it down.
    if (activeTTS) { activeTTS.cancel(); activeTTS = null; }

    appendMessage('user', userText);
    history.push({ role: 'user', content: userText });
    pruneHistory();
    textInput.value = '';

    const botDiv = appendMessage('bot', '');
    botDiv.classList.add('streaming');

    const wsUrl = `ws://${window.location.host}/ws/chat`;
    const ws = new WebSocket(wsUrl);
    let botText = '';      // displayed text (raw stream)
    let ttsBuffer = '';    // unflushed tail still waiting for a sentence terminator
    let finished = false;

    setStatus(currentLang === 'nl' ? 'Denkt na…' : 'Thinking…', true);

    const tts = new TTSPlayer(currentLang, () => {
      // First chunk has started playing — switch status from "Thinking…" to "Speaking…".
      setStatus(currentLang === 'nl' ? 'Spreekt…' : 'Speaking…', true);
    });
    activeTTS = tts;

    function tryFlushTTS() {
      const flushable = extractFlushableSentence(ttsBuffer);
      if (!flushable) return;
      ttsBuffer = ttsBuffer.slice(flushable.length);
      const cleaned = cleanForTTS(stripBracketedTranslations(flushable));
      if (cleaned) tts.enqueue(cleaned);
    }

    ws.onopen = () => {
      // Server reads history from disk — we only send the new message + language.
      const payload = {
        message: userText,
        lang: currentLang,
      };
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (e) => {
      if (e.data === '[DONE]') {
        finished = true;
        // Flush any remaining tail (no terminator) so it gets spoken too.
        const tail = ttsBuffer.trim();
        ttsBuffer = '';
        if (tail) {
          const cleaned = cleanForTTS(stripBracketedTranslations(tail));
          if (cleaned) tts.enqueue(cleaned);
        }

        const finalText = stripBracketedTranslations(botText);
        botDiv.textContent = finalText;
        botDiv.classList.remove('streaming');
        history.push({ role: 'assistant', content: finalText });
        pruneHistory();

        // Stay busy until audio queue drains, so VAD-driven auto-sends can't trip mid-playback.
        tts.onIdle(() => {
          if (activeTTS === tts) activeTTS = null;
          clearStatus();
        });
        return;
      }
      if (e.data.startsWith('[ERROR]')) {
        finished = true;
        ws.close();
        tts.cancel();
        if (activeTTS === tts) activeTTS = null;
        botDiv.classList.remove('streaming');
        botDiv.textContent = e.data;
        statusDot.className = 'error';
        setStatus('Error from LLM.');
        return;
      }
      botText   += e.data;
      ttsBuffer += e.data;
      botDiv.textContent = botText;
      chat.scrollTop = chat.scrollHeight;

      tryFlushTTS();
    };

    ws.onerror = () => {
      if (finished) return;
      tts.cancel();
      if (activeTTS === tts) activeTTS = null;
      botDiv.classList.remove('streaming');
      botDiv.textContent = '[Connection error]';
      statusDot.className = 'error';
      setStatus('WebSocket error.');
    };
  }

  sendBtn.addEventListener('click', () => sendMessage(textInput.value));
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(textInput.value);
    }
  });

  // ── Microphone: arm/disarm + VAD-driven recording ────────────────────────
  micBtn.addEventListener('click', async () => {
    if (micArmed) {
      disarmMic();
    } else {
      await armMic();
    }
  });

  async function armMic() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Disable browser DSP — it would suppress the silence/noise jump we use as the trigger.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC();
      const src = audioCtx.createMediaStreamSource(micStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.2;
      src.connect(analyser);

      micArmed = true;
      recState = 'idle';
      aboveSinceMs = 0;
      belowSinceMs = 0;
      micBtn.classList.add('armed');
      showRmsMeter(true);
      setStatus(currentLang === 'nl'
        ? 'Microfoon gewapend — zet de podiummicrofoon aan om te spreken.'
        : 'Mic armed — turn the stage mic on to speak.');
      vadLoop();
    } catch (e) {
      console.error('Mic error:', e);
      setStatus('Microphone access denied.');
    }
  }

  function disarmMic() {
    micArmed = false;
    if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = null; }
    if (recState === 'recording') {
      try { mediaRecorder.stop(); } catch (_) {}
      recState = 'idle';
    }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
    analyser = null;
    micBtn.classList.remove('armed', 'recording');
    showRmsMeter(false);
    clearStatus();
  }

  function showRmsMeter(visible) {
    rmsMeter.hidden = !visible;
    if (visible) {
      const pctOpen  = Math.min(100, (VAD.OPEN_THRESHOLD  / RMS_DISPLAY_MAX) * 100);
      const pctClose = Math.min(100, (VAD.CLOSE_THRESHOLD / RMS_DISPLAY_MAX) * 100);
      rmsMarkOpen.style.left  = pctOpen  + '%';
      rmsMarkClose.style.left = pctClose + '%';
    } else {
      rmsFill.style.width = '0%';
      rmsValue.textContent = '0.000';
    }
  }

  function updateRmsMeter(rms) {
    const pct = Math.min(100, (rms / RMS_DISPLAY_MAX) * 100);
    rmsFill.style.width = pct + '%';
    rmsValue.textContent = rms.toFixed(3);
  }

  function computeRMS(buf) {
    // buf is Uint8 time-domain data, 0..255 with 128 = silence
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / buf.length);
  }

  function vadLoop() {
    if (!micArmed || !analyser) return;
    const buf = new Uint8Array(analyser.fftSize);
    const tick = () => {
      if (!micArmed || !analyser) return;
      analyser.getByteTimeDomainData(buf);
      const rms = computeRMS(buf);
      updateRmsMeter(rms);
      const now = performance.now();

      if (recState === 'idle') {
        if (rms >= VAD.OPEN_THRESHOLD) {
          if (!aboveSinceMs) aboveSinceMs = now;
          if (now - aboveSinceMs >= VAD.OPEN_SUSTAIN_MS) {
            startRecorder();
          }
        } else {
          aboveSinceMs = 0;
        }
      } else { // recording
        if (rms <= VAD.CLOSE_THRESHOLD) {
          if (!belowSinceMs) belowSinceMs = now;
          if (now - belowSinceMs >= VAD.CLOSE_SUSTAIN_MS) {
            stopRecorder();
          }
        } else {
          belowSinceMs = 0;
        }
      }

      rafHandle = requestAnimationFrame(tick);
    };
    rafHandle = requestAnimationFrame(tick);
  }

  function startRecorder() {
    if (!micStream) return;
    audioChunks = [];
    aboveSinceMs = 0;
    belowSinceMs = 0;
    recordStartMs = performance.now();
    try {
      mediaRecorder = new MediaRecorder(micStream);
    } catch (e) {
      console.error('MediaRecorder unsupported:', e);
      setStatus('Recorder unsupported in this browser.');
      return;
    }
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const durationMs = performance.now() - recordStartMs;
      if (durationMs < VAD.MIN_UTTERANCE_MS || audioChunks.length === 0) {
        // Too short — likely a noise blip. Drop it and re-arm.
        if (micArmed) setStatus(currentLang === 'nl'
          ? 'Microfoon gewapend — zet de podiummicrofoon aan om te spreken.'
          : 'Mic armed — turn the stage mic on to speak.');
        return;
      }
      transcribeAudio();
    };
    mediaRecorder.start();
    recState = 'recording';
    micBtn.classList.add('recording');
    setStatus(currentLang === 'nl' ? 'Luistert…' : 'Listening…', true);
  }

  function stopRecorder() {
    if (recState !== 'recording') return;
    try { mediaRecorder.stop(); } catch (_) {}
    recState = 'idle';
    aboveSinceMs = 0;
    belowSinceMs = 0;
    micBtn.classList.remove('recording');
  }

  async function transcribeAudio() {
    setStatus(currentLang === 'nl' ? 'Transcribeert…' : 'Transcribing…', true);
    try {
      const mimeType = audioChunks[0]?.type || 'audio/webm';
      const blob = new Blob(audioChunks, { type: mimeType });
      const ext = mimeType.includes('ogg') ? '.ogg' : mimeType.includes('mp4') ? '.mp4' : '.webm';

      const form = new FormData();
      form.append('audio', blob, `recording${ext}`);

      const res = await fetch('/transcribe', { method: 'POST', body: form });
      const data = await res.json();

      if (data.error) throw new Error(data.error);
      if (data.text && data.text.trim()) {
        if (VAD.AUTO_SEND) {
          sendMessage(data.text.trim());
        } else {
          textInput.value = data.text.trim();
          textInput.focus();
          clearStatus();
        }
      } else {
        clearStatus();
      }
    } catch (e) {
      console.error('Transcription error:', e);
      setStatus('Transcription failed: ' + e.message);
    } finally {
      // Re-show armed status once any auto-send completes; if AUTO_SEND triggered sendMessage,
      // it will overwrite the status with "Thinking…" already, so this only matters on errors/empties.
      if (micArmed && !isBusy) {
        setStatus(currentLang === 'nl'
          ? 'Microfoon gewapend — zet de podiummicrofoon aan om te spreken.'
          : 'Mic armed — turn the stage mic on to speak.');
      }
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  syncLang();
  loadHistoryFromServer();
})();
