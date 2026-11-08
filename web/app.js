import { AutoScrollController, SelectionTranslator, messageIsFinal } from './ui-interactions.js';

(() => {
  'use strict';

  const languages = [
    ['AR', 'Arabic', 'العربية'], ['HE', 'Hebrew', 'עברית'],
    ['DE', 'German', 'Deutsch'], ['TR', 'Turkish', 'Türkçe'],
    ['FR', 'French', 'Français'], ['RU', 'Russian', 'Русский'],
    ['EN', 'English', 'English'], ['ES', 'Spanish', 'Español'],
  ];
  const languageNames = Object.fromEntries(languages.map(([code, name]) => [code, name]));
  const $ = id => document.getElementById(id);
  const elements = Object.fromEntries([
    'connection', 'recording-pill', 'session-state', 'session-form', 'source-language',
    'target-language', 'input-device', 'refresh-devices', 'swap-languages', 'debug-recording',
    'debug-recording-label', 'recording-explainer',
    'start-session', 'stop-session', 'review-clip', 'error-banner', 'error-text', 'dismiss-error',
    'incoming-pair', 'reply-pair', 'transcript-scroll', 'transcript-list', 'transcript-empty',
    'replies-scroll', 'replies-list', 'replies-empty', 'reply-form', 'reply-text', 'send-reply',
    'composer-label', 'composer-help', 'reply-destination', 'runtime-status', 'detail-session',
    'detail-recording', 'detail-directory', 'flush-audio', 'input-hint', 'toast',
    'transcript-autoscroll', 'replies-autoscroll', 'selection-tooltip',
  ].map(id => [id, $(id)]));

  let state = null;
  let socket = null;
  let connected = false;
  let settingsDirty = false;
  let composing = false;
  let reconnectDelay = 500;
  let reconnectTimer;
  let toastTimer;
  let previousSession = null;
  let lastVersion = -1;
  let lastConfig = '';
  const pending = new Set();
  const messageNodes = { transcript: new Map(), replies: new Map() };
  const scrollControllers = {
    transcript: new AutoScrollController(elements['transcript-scroll'], elements['transcript-autoscroll']),
    replies: new AutoScrollController(elements['replies-scroll'], elements['replies-autoscroll']),
  };

  const codeOf = value => {
    const code = String(value || '').toUpperCase().split(/[-_]/)[0];
    return { ARA: 'AR', ARB: 'AR', HEB: 'HE', GER: 'DE', DEU: 'DE', TUR: 'TR', FRE: 'FR', FRA: 'FR', RUS: 'RU', ENG: 'EN', SPA: 'ES', SP: 'ES' }[code] || code;
  };
  const nameOf = code => languageNames[codeOf(code)] || String(code || 'Language');
  const directionOf = code => ['AR', 'HE'].includes(codeOf(code)) ? 'rtl' : 'ltr';
  const cleanDisplay = value => String(value ?? '').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
  const selectionTranslator = new SelectionTranslator({
    tooltip: elements['selection-tooltip'],
    getSession: () => state?.controller_run_id || state?.session_id || 'stopped',
    nameOf,
    onSelection: block => scrollControllers[block.closest('.incoming-panel') ? 'transcript' : 'replies'].pause(),
    onRelease: () => queueMicrotask(() => { if (state) applyState(state); }),
  });

  for (const id of ['source-language', 'target-language']) {
    for (const [code, name, nativeName] of languages) {
      const option = document.createElement('option');
      option.value = code;
      option.textContent = `${name} · ${nativeName}`;
      elements[id].append(option);
    }
  }
  elements['source-language'].value = 'TR';
  elements['target-language'].value = 'EN';

  function setConnection(kind, label) {
    connected = kind === 'connected';
    elements.connection.dataset.state = kind;
    elements.connection.querySelector('span').textContent = label;
    renderControls();
  }

  function showError(message) {
    elements['error-text'].textContent = cleanDisplay(message);
    elements['error-banner'].hidden = false;
  }

  function showToast(message, isError = false) {
    clearTimeout(toastTimer);
    elements.toast.textContent = cleanDisplay(message);
    elements.toast.dataset.error = String(isError);
    elements.toast.hidden = false;
    toastTimer = setTimeout(() => { elements.toast.hidden = true; }, 4000);
  }

  async function api(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? { Accept: 'application/json' } : { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: 'same-origin',
        cache: 'no-store',
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false || payload.accepted === false) {
        throw new Error(payload.detail || payload.error || payload.message || `Request failed (${response.status})`);
      }
      return payload;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The local service did not respond. Your draft is kept.');
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function refreshState() {
    const observedVersion = lastVersion;
    return api('/api/state').then(value => {
      // An earlier HTTP snapshot must not overwrite a newer live update.
      if (lastVersion === observedVersion) applyState(value);
    });
  }

  async function command(name, body = {}) {
    if (pending.has(name)) return null;
    pending.add(name);
    renderControls();
    try {
      const result = await api(`/api/${name}`, body);
      if (result.state && typeof result.state === 'object') applyState(result.state);
      return result;
    } catch (error) {
      showError(error.message || 'The request could not be completed.');
      return null;
    } finally {
      pending.delete(name);
      renderControls();
    }
  }

  function stageInfo(value) {
    if (typeof value === 'string') {
      const parts = value.split('|');
      return { state: parts.shift().trim().toLowerCase(), detail: parts.join(' | ').trim() };
    }
    if (value && typeof value === 'object') {
      return {
        state: String(value.state || value.status || 'idle').toLowerCase(),
        detail: cleanDisplay(value.detail || value.error || value.message || ''),
      };
    }
    return { state: 'idle', detail: '' };
  }

  function renderHealth(stage, fallback) {
    const info = stageInfo(state?.stages?.[stage]);
    const node = $(`health-${stage}`);
    const known = ['idle', 'off', 'new', 'loading', 'ready', 'busy', 'failed', 'closed', 'listening', 'no input', 'starting', 'stopping'];
    const key = known.includes(info.state) ? info.state : 'idle';
    const labels = { off: 'Off', new: 'Starting', idle: 'Idle', closed: 'Stopped', 'no input': 'No input' };
    node.dataset.state = key;
    node.querySelector('.health-state').textContent = labels[key] || key.charAt(0).toUpperCase() + key.slice(1);
    node.querySelector('.health-detail').textContent = info.detail || fallback;
    node.querySelector('.health-detail').title = info.detail || fallback;
  }

  function renderControls() {
    const running = Boolean(state?.running);
    const changing = pending.has('start') || pending.has('stop');
    for (const id of ['source-language', 'target-language', 'input-device', 'swap-languages', 'debug-recording']) {
      elements[id].disabled = running || changing;
    }
    elements['refresh-devices'].disabled = pending.has('devices');
    elements['start-session'].hidden = running;
    elements['stop-session'].hidden = !running;
    elements['start-session'].disabled = !connected || changing;
    elements['stop-session'].disabled = !connected || changing;
    elements['start-session'].lastChild.textContent = pending.has('start') ? ' Starting…' : 'Start listening';
    elements['stop-session'].lastChild.textContent = pending.has('stop') ? ' Stopping…' : 'Stop session';
    elements['flush-audio'].disabled = !connected || !running || pending.has('flush');
    elements['review-clip'].disabled = !connected || !running || pending.has('mark');
    const failed = stageInfo(state?.stages?.translation).state === 'failed';
    elements['send-reply'].disabled = !connected || !running || failed || pending.has('reply') || !elements['reply-text'].value.trim();
    elements['reply-text'].disabled = false;
    if (!connected) elements['composer-help'].textContent = 'Reconnecting to the service · your draft is kept';
    else if (!running) elements['composer-help'].textContent = 'Start a session to translate and speak your reply';
    else if (failed) elements['composer-help'].textContent = 'Translation is unavailable · your draft is kept';
    else elements['composer-help'].textContent = 'Enter to send · Shift + Enter for a new line';
  }

  function updateLanguageLabels() {
    const config = state?.config || {};
    const source = codeOf(config.source_language || elements['source-language'].value || 'TR');
    const target = codeOf(config.target_language || elements['target-language'].value || 'EN');
    elements['incoming-pair'].textContent = `${nameOf(source)} → ${nameOf(target)}`;
    elements['reply-pair'].textContent = `${nameOf(target)} → ${nameOf(source)}`;
    elements['composer-label'].textContent = `Write in ${nameOf(target)}`;
    elements['reply-destination'].textContent = `Spoken in ${nameOf(source)}`;
    elements['reply-text'].lang = target.toLowerCase();
    elements['reply-text'].dir = 'auto';
    elements['reply-text'].parentElement.dir = directionOf(target);
    return { source, target };
  }

  function buildMessage() {
    const card = document.createElement('article');
    card.className = 'message-card';
    const topline = document.createElement('div');
    topline.className = 'message-topline';
    const number = document.createElement('span');
    number.className = 'message-number';
    const phase = document.createElement('span');
    phase.className = 'message-phase';
    topline.append(number, phase);
    card.append(topline);
    const parts = {};
    for (const kind of ['source', 'translation']) {
      const part = document.createElement('div');
      part.className = `message-part ${kind}`;
      const label = document.createElement('span');
      label.className = 'part-label';
      const content = document.createElement('bdi');
      content.className = 'message-text';
      content.dir = 'auto';
      part.append(label, content);
      card.append(part);
      parts[kind] = { part, label, content };
    }
    return { card, number, phase, parts };
  }

  function renderMessages(kind, messages, source, target) {
    const scroll = elements[`${kind === 'transcript' ? 'transcript' : 'replies'}-scroll`];
    const list = elements[`${kind === 'transcript' ? 'transcript' : 'replies'}-list`];
    const empty = elements[`${kind === 'transcript' ? 'transcript' : 'replies'}-empty`];
    const scrollSnapshot = scrollControllers[kind].beforeRender(list);
    const protectedCard = selectionTranslator.protectedCard();
    const records = Array.isArray(messages) ? messages : [];
    const nodes = messageNodes[kind];
    const seen = new Set(records.map((item, index) => String(item?.id ?? item?.utterance_id ?? `message-${index}`)));
    // Prune before ordering so bounded-history removal does not reparent the
    // selected card. Keep a selected old card until the user releases it.
    for (const [key, node] of nodes) {
      if (!seen.has(key) && node.card !== protectedCard) { node.card.remove(); nodes.delete(key); }
    }
    for (let index = 0; index < records.length; index += 1) {
      const item = records[index] || {};
      const key = String(item.id ?? item.utterance_id ?? `message-${index}`);
      let node = nodes.get(key);
      if (!node) { node = buildMessage(); nodes.set(key, node); }
      if (protectedCard) {
        if (!node.card.isConnected) list.append(node.card);
      } else if (list.children[index] !== node.card) list.insertBefore(node.card, list.children[index] || null);
      if (node.card === protectedCard) continue;
      node.number.textContent = `${kind === 'transcript' ? 'PHRASE' : 'REPLY'} ${String(index + 1).padStart(2, '0')}`;
      const sourceFinal = item.source_final ?? item.src_final ?? false;
      const translationFinal = item.translation_final ?? item.dst_final ?? false;
      node.card.dataset.final = String(messageIsFinal(item));
      node.phase.dataset.final = String(Boolean(sourceFinal && translationFinal));
      node.phase.textContent = sourceFinal && translationFinal ? 'Complete' : sourceFinal ? 'Translating' : kind === 'transcript' ? 'Listening' : 'Translating';
      const sourceCode = codeOf(item.source_language || source);
      const targetCode = codeOf(item.target_language || target);
      for (const [partKind, text, language] of [
        ['source', item.source ?? item.src ?? '', sourceCode],
        ['translation', item.translation ?? item.dst ?? '', targetCode],
      ]) {
        const part = node.parts[partKind];
        const clean = cleanDisplay(text);
        const placeholder = !clean;
        part.part.dir = directionOf(language);
        part.content.lang = language.toLowerCase();
        part.content.dir = 'auto';
        part.content.dataset.sourceLanguage = language;
        part.content.dataset.targetLanguage = partKind === 'source' ? targetCode : sourceCode;
        part.label.textContent = `${nameOf(language).toUpperCase()}${partKind === 'translation' ? ' · TRANSLATION' : ' · ORIGINAL'}`;
        const display = clean || (partKind === 'source' ? 'Listening…' : stageInfo(state?.stages?.translation).state === 'failed' ? 'Translation unavailable' : 'Translation will appear here…');
        if (part.content.textContent !== display) part.content.textContent = display;
        part.content.classList.toggle('placeholder', placeholder);
      }
    }
    empty.hidden = records.length > 0;
    scrollControllers[kind].afterRender(scrollSnapshot);
  }

  function applyState(next, version) {
    if (!next || typeof next !== 'object') return;
    const session = next.session_id ?? null;
    if (Number.isFinite(version) && version < lastVersion && session === previousSession) return;
    if (Number.isFinite(version)) lastVersion = version;
    if (session !== previousSession) {
      selectionTranslator.reset();
      for (const group of Object.values(messageNodes)) {
        for (const node of group.values()) node.card.remove();
        group.clear();
      }
      previousSession = session;
      lastVersion = Number.isFinite(version) ? version : -1;
    }
    state = next;
    const config = state.config || {};
    const reviewPolicy = config.audio_review || 'auto';
    elements['debug-recording-label'].textContent = reviewPolicy === 'auto' ? 'Debug · record full audio' : 'Debug diagnostics';
    elements['recording-explainer'].textContent = {
      auto: 'Otherwise, only flagged audio clips are saved for review.',
      full: 'Full audio recording is enabled for this launch.',
      issues: 'Only flagged audio clips are saved for this launch.',
      off: 'Audio recording is disabled for this launch.',
    }[reviewPolicy] || '';
    const configKey = JSON.stringify([config.source_language, config.target_language, config.input_device, config.debug]);
    if ((!settingsDirty || state.running) && configKey !== lastConfig) {
      elements['source-language'].value = codeOf(config.source_language || 'TR');
      elements['target-language'].value = codeOf(config.target_language || 'EN');
      const deviceValue = config.input_device == null ? '' : String(config.input_device);
      if (![...elements['input-device'].options].some(option => option.value === deviceValue)) {
        const option = new Option(deviceValue, deviceValue);
        elements['input-device'].add(option);
      }
      elements['input-device'].value = deviceValue;
      elements['debug-recording'].checked = Boolean(config.debug);
      settingsDirty = false;
      lastConfig = configKey;
    }
    const { source, target } = updateLanguageLabels();
    document.body.classList.toggle('running', Boolean(state.running));
    elements['session-state'].textContent = state.running ? 'Session in progress' : 'Ready when you are';
    elements['session-state'].dataset.running = String(Boolean(state.running));
    renderHealth('capture', state.running ? 'Waiting for incoming audio' : 'Not listening');
    renderHealth('asr', state.running ? 'Preparing transcription' : 'Waiting for a session');
    renderHealth('translation', state.running ? 'Preparing translation' : 'Waiting for a session');
    renderMessages('transcript', state.transcript, source, target);
    renderMessages('replies', state.replies, target, source);
    elements['runtime-status'].textContent = cleanDisplay(state.status) || (state.running ? 'Listening and processing independently. You can reply at any time.' : 'Choose your languages and audio input to begin.');
    elements['detail-session'].textContent = state.session_id || 'Not started';
    const mode = state.audio_review_mode || (config.debug ? 'full' : 'issues');
    const modeLabel = { full: 'Full audio recording', issues: 'Issue clips only', off: 'Audio recording off', auto: config.debug ? 'Full audio recording' : 'Issue clips only' }[mode] || mode;
    elements['recording-pill'].hidden = !state.running;
    elements['recording-pill'].dataset.mode = mode;
    elements['recording-pill'].textContent = modeLabel;
    elements['detail-recording'].textContent = modeLabel;
    elements['detail-directory'].textContent = state.recording_directory || '—';
    const capture = stageInfo(state.stages?.capture);
    elements['input-hint'].replaceChildren();
    const dot = document.createElement('span');
    dot.className = 'footer-dot';
    dot.setAttribute('aria-hidden', 'true');
    elements['input-hint'].append(dot, document.createTextNode(state.running ? capture.state === 'no input' ? 'No incoming audio · check the input route' : 'Audio capture stays active while you reply' : 'Listening starts when your session begins'));
    renderControls();
  }

  async function refreshDevices() {
    if (pending.has('devices')) return;
    pending.add('devices');
    renderControls();
    try {
      const payload = await api('/api/devices');
      const selected = elements['input-device'].value;
      elements['input-device'].replaceChildren(new Option('System default input', ''));
      for (const device of payload.devices || []) {
        const value = device.id ?? device.index ?? device.name;
        if (value == null || device.max_input_channels === 0) continue;
        const option = new Option(device.name || `Input ${value}`, String(value));
        elements['input-device'].add(option);
      }
      if (selected && ![...elements['input-device'].options].some(option => option.value === selected)) {
        elements['input-device'].add(new Option(selected, selected));
      }
      elements['input-device'].value = selected;
      if (payload.error) showError(payload.error);
    } catch (error) {
      showError(`Audio inputs could not be listed: ${error.message}`);
    } finally {
      pending.delete('devices');
      renderControls();
    }
  }

  function connect() {
    clearTimeout(reconnectTimer);
    setConnection('connecting', 'Connecting');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
    socket.addEventListener('open', () => {
      reconnectDelay = 500;
      lastVersion = -1;
      setConnection('connected', 'Connected locally');
      refreshState().catch(() => {});
    });
    socket.addEventListener('message', event => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'error') { showError(payload.message || payload.error || 'Service error'); return; }
        if (payload.type === 'state' || !payload.type) applyState(payload.state || payload, payload.version);
      } catch (_) { /* Ignore malformed/unrelated messages; reconnect keeps the draft. */ }
    });
    socket.addEventListener('close', () => {
      setConnection('disconnected', 'Reconnecting');
      reconnectTimer = setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 1.7, 10000);
    });
    socket.addEventListener('error', () => socket.close());
  }

  elements['session-form'].addEventListener('submit', async event => {
    event.preventDefault();
    const source = elements['source-language'].value;
    const target = elements['target-language'].value;
    if (source === target) { showError('Choose different incoming and translation languages.'); return; }
    const rawDevice = elements['input-device'].value;
    const inputDevice = rawDevice === '' ? null : /^\d+$/.test(rawDevice) ? Number(rawDevice) : rawDevice;
    const result = await command('start', { source_language: source, target_language: target, input_device: inputDevice, debug: elements['debug-recording'].checked });
    if (result) {
      settingsDirty = false;
      elements['error-banner'].hidden = true;
      showToast('Session starting. You can type while models load.');
      refreshState().catch(() => {});
    }
  });
  elements['stop-session'].addEventListener('click', async () => {
    const result = await command('stop');
    if (result) showToast('Session stopped. Your draft is kept.');
  });
  elements['flush-audio'].addEventListener('click', async () => {
    if (await command('flush')) showToast('Finishing the current phrase.');
  });
  elements['review-clip'].addEventListener('click', async () => {
    const result = await command('mark');
    if (result) showToast(result.message || 'Audio marked for review.');
  });
  elements['refresh-devices'].addEventListener('click', refreshDevices);
  elements['dismiss-error'].addEventListener('click', () => { elements['error-banner'].hidden = true; });
  elements['swap-languages'].addEventListener('click', () => {
    const source = elements['source-language'].value;
    elements['source-language'].value = elements['target-language'].value;
    elements['target-language'].value = source;
    settingsDirty = true;
  });
  for (const id of ['source-language', 'target-language', 'input-device', 'debug-recording']) {
    elements[id].addEventListener('change', () => { settingsDirty = true; });
  }

  async function sendReply(event) {
    if (event) event.preventDefault();
    if (elements['send-reply'].disabled || composing) return;
    const draft = elements['reply-text'].value;
    if (!draft.trim()) return;
    const result = await command('reply', { text: draft });
    if (result && elements['reply-text'].value === draft) elements['reply-text'].value = '';
    renderControls();
    elements['reply-text'].focus({ preventScroll: true });
  }
  elements['reply-form'].addEventListener('submit', sendReply);
  elements['reply-text'].addEventListener('input', renderControls);
  elements['reply-text'].addEventListener('compositionstart', () => { composing = true; });
  elements['reply-text'].addEventListener('compositionend', () => { composing = false; renderControls(); });
  elements['reply-text'].addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) {
      event.preventDefault();
      sendReply();
    }
  });
  window.addEventListener('online', () => {
    if (!connected && (!socket || socket.readyState === WebSocket.CLOSED)) connect();
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && connected) refreshState().catch(() => {});
  });

  renderControls();
  if (location.protocol === 'file:') {
    applyState({ running: false, config: { source_language: 'TR', target_language: 'EN', debug: false },
      transcript: [], replies: [], stages: { capture: 'off', asr: 'off', translation: 'off' },
      status: 'Offline preview. Run python tts-transcribe.py --ui web and open its local URL to connect.' });
    setConnection('disconnected', 'Offline preview');
  } else {
    refreshDevices();
    connect();
  }
})();
