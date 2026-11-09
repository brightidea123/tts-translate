/* Browser-only interactions stay independent of the audio/model controller. */

export function inputSelectionAvailable(select) {
  return [...select.options].some(option => option.value === select.value && !option.disabled);
}

export function selectInputDevice(select, value, OptionClass = Option) {
  if (![...select.options].some(option => option.value === value)) {
    const option = new OptionClass(`Unavailable input: ${value} — choose an input`, value);
    option.disabled = true;
    select.add(option);
  }
  select.value = value;
}

export function refreshInputOptions(select, devices, OptionClass = Option) {
  const selected = select.value;
  select.replaceChildren(new OptionClass('System default input', ''));
  for (const device of devices) {
    if (!device.name || device.max_input_channels < 1) continue;
    // Names survive PortAudio renumbering. Never turn an old index into a
    // different source, or silently replace a missing LitLink input with a mic.
    const label = device.ambiguous ? `${device.name} — ambiguous input name` : device.name;
    const option = new OptionClass(label, device.name);
    option.disabled = Boolean(device.ambiguous);
    select.add(option);
  }
  selectInputDevice(select, selected, OptionClass);
}

export class BoundedCache {
  constructor(limit = 64) { this.limit = limit; this.values = new Map(); }
  get(key) {
    if (!this.values.has(key)) return undefined;
    const value = this.values.get(key);
    this.values.delete(key);
    this.values.set(key, value);
    return value;
  }
  set(key, value) {
    this.values.delete(key);
    this.values.set(key, value);
    while (this.values.size > this.limit) this.values.delete(this.values.keys().next().value);
  }
}

export function selectionLanguages(block) {
  const source = block?.dataset?.sourceLanguage;
  const target = block?.dataset?.targetLanguage;
  return source && target && source !== target ? { source_language: source, target_language: target } : null;
}

export function messageIsFinal(item) {
  return Boolean((item.source_final ?? item.src_final ?? false) && (item.translation_final ?? item.dst_final ?? false));
}

export function messageStatus(item, kind = 'transcript') {
  const reviewRequired = item.review_required === true;
  const sourceFinal = item.source_final ?? item.src_final ?? false;
  const final = messageIsFinal(item);
  return {
    complete: reviewRequired || final,
    reviewRequired,
    label: reviewRequired ? 'Review needed' : final ? 'Complete' : sourceFinal ? 'Translating' : kind === 'transcript' ? 'Listening' : 'Translating',
    reason: reviewRequired ? String(item.review_reason || 'Final recognition needs review') : '',
  };
}

const phraseTimeFormatters = new BoundedCache(8);

function phraseTimeFormats(timeZone) {
  const key = timeZone ?? null;
  let formats = phraseTimeFormatters.get(key);
  if (!formats) {
    formats = {
      clock: new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23', timeZone }),
      full: new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'long', timeZone }),
    };
    phraseTimeFormatters.set(key, formats);
  }
  return formats;
}

export function formatPhraseTiming(item, { timeZone } = {}) {
  const timestamp = typeof item?.audio_started_at === 'string' ? Date.parse(item.audio_started_at) : NaN;
  const duration = item?.audio_duration_seconds;
  const parts = [], details = [];
  if (Number.isFinite(timestamp)) {
    const date = new Date(timestamp);
    const formats = phraseTimeFormats(timeZone);
    parts.push(formats.clock.format(date));
    details.push(`Speech clip starts at ${formats.full.format(date)}`);
  }
  if (typeof duration === 'number' && Number.isFinite(duration) && duration >= 0) {
    parts.push(`${duration.toFixed(1)} s`);
    details.push(`Speech clip duration: ${duration.toFixed(1)} seconds`);
  }
  return { text: parts.join(' · '), title: details.join(' · ') };
}

const phraseTimingRenders = new WeakMap();

export function renderPhraseTiming(element, item, { timeZone } = {}) {
  const startedAt = item?.audio_started_at, duration = item?.audio_duration_seconds;
  const previous = phraseTimingRenders.get(element);
  // Model/source updates do not change a finalized clip's timing. Leave its
  // label untouched and avoid date parsing/formatting on every state snapshot.
  if (previous && previous.startedAt === startedAt && Object.is(previous.duration, duration) && previous.timeZone === timeZone) return previous.value;
  const value = formatPhraseTiming(item, { timeZone });
  element.textContent = value.text;
  element.title = value.title;
  element.setAttribute('aria-label', value.title);
  element.hidden = !value.text;
  phraseTimingRenders.set(element, { startedAt, duration, timeZone, value });
  return value;
}

export function sourcePreviewSegments(item, source) {
  const final = Boolean(item.source_final ?? item.src_final ?? false);
  const text = String(source ?? '');
  const placeholder = !text || (!final && text === '..');
  if (placeholder) return { text: '..', stable: '', provisional: '', placeholder: true };
  if (final) return { text, stable: text, provisional: '', placeholder: false };
  const candidate = typeof item.source_stable === 'string' ? item.source_stable : '';
  const stable = text.startsWith(candidate) ? candidate : '';
  return { text, stable, provisional: text.slice(stable.length), placeholder: false };
}

const sourcePreviewRenders = new WeakMap();

export function renderSourcePreview(content, item, source, documentObject = document) {
  const preview = sourcePreviewSegments(item, source);
  // Busy status changes must not replace already-visible words or disturb
  // selection. Only a changed reading or changed stability needs new nodes.
  const key = JSON.stringify([preview.text, preview.stable, preview.placeholder]);
  const previous = sourcePreviewRenders.get(content);
  if (previous?.key === key) return preview;
  if (previous && !previous.placeholder && !preview.placeholder && previous.stable === preview.stable) {
    // Keep the steady prefix node in place as only the uncertain ending is
    // revised. Readers and selections in that prefix retain their DOM anchor.
    if (preview.provisional) {
      if (!previous.tail) {
        previous.tail = documentObject.createElement('span');
        previous.tail.className = 'transcript-provisional';
        content.append(previous.tail);
      }
      previous.tail.textContent = preview.provisional;
    } else if (previous.tail) {
      content.removeChild(previous.tail);
      previous.tail = null;
    }
    previous.key = key;
    return preview;
  }
  const nodes = [];
  let tail = null;
  if (preview.placeholder) nodes.push(documentObject.createTextNode('..'));
  else {
    if (preview.stable) nodes.push(documentObject.createTextNode(preview.stable));
    if (preview.provisional) {
      tail = documentObject.createElement('span');
      tail.className = 'transcript-provisional';
      tail.textContent = preview.provisional;
      nodes.push(tail);
    }
  }
  content.replaceChildren(...nodes);
  content.classList.toggle('placeholder', preview.placeholder);
  sourcePreviewRenders.set(content, { key, stable: preview.stable, placeholder: preview.placeholder, tail });
  return preview;
}

export class AutoScrollController {
  constructor(scroller, checkbox) {
    this.scroller = scroller;
    this.checkbox = checkbox;
    this.lastTop = scroller.scrollTop;
    this.touchY = null;
    scroller.addEventListener('wheel', event => { if (event.deltaY < 0) this.pause(); }, { passive: true });
    scroller.addEventListener('touchstart', event => { this.touchY = event.touches[0]?.clientY ?? null; }, { passive: true });
    scroller.addEventListener('touchmove', event => {
      const y = event.touches[0]?.clientY;
      if (this.touchY !== null && y > this.touchY + 2) this.pause();
      this.touchY = y ?? this.touchY;
    }, { passive: true });
    scroller.addEventListener('keydown', event => {
      if (['ArrowUp', 'PageUp', 'Home'].includes(event.key) || (event.key === ' ' && event.shiftKey)) this.pause();
    });
    // Scroll also covers scrollbar dragging and assistive input. Programmatic
    // writes update lastTop synchronously before the browser's scroll event.
    scroller.addEventListener('scroll', () => {
      if (scroller.scrollTop < this.lastTop - 2) this.pause();
      this.lastTop = scroller.scrollTop;
    }, { passive: true });
    checkbox.addEventListener('change', () => { if (checkbox.checked) this.toBottom(); });
  }
  pause() { this.checkbox.checked = false; }
  writeTop(top) {
    this.scroller.scrollTop = top;
    this.lastTop = this.scroller.scrollTop;
  }
  toBottom() { this.writeTop(this.scroller.scrollHeight); }
  beforeRender(list) {
    if (this.checkbox.checked) return null;
    const viewport = this.scroller.getBoundingClientRect();
    const anchors = [...list.children].map(node => ({ node, top: node.getBoundingClientRect().top }))
      .filter(anchor => anchor.node.getBoundingClientRect().bottom > viewport.top && anchor.top < viewport.bottom);
    return { top: this.scroller.scrollTop, anchors };
  }
  afterRender(snapshot) {
    if (this.checkbox.checked) { this.toBottom(); return; }
    // Preserve the first surviving visible card when bounded history removes
    // earlier cards. Appending a new phrase never moves someone reading history.
    const anchor = snapshot?.anchors.find(item => item.node.isConnected);
    const delta = anchor ? anchor.node.getBoundingClientRect().top - anchor.top : 0;
    this.writeTop((anchor ? this.scroller.scrollTop : snapshot?.top ?? this.scroller.scrollTop) + delta);
  }
}

export function selectedMessage(selection) {
  if (!selection || selection.isCollapsed || selection.rangeCount !== 1) return null;
  const element = node => node?.nodeType === 1 ? node : node?.parentElement;
  const start = element(selection.anchorNode)?.closest('.message-text');
  const end = element(selection.focusNode)?.closest('.message-text');
  if (!start || start !== end || start.classList.contains('placeholder')) return null;
  const languages = selectionLanguages(start);
  const text = String(selection.toString()).trim();
  if (!languages || !text) return null;
  return { block: start, text, ...languages, range: selection.getRangeAt(0) };
}

export function abortableDelay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException('Selection changed', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, milliseconds);
    if (signal.aborted) { abort(); return; }
    signal.addEventListener('abort', abort, { once: true });
  });
}

export class SelectionTranslator {
  constructor({ tooltip, getSession, nameOf, onSelection, onRelease, documentObject = document,
                windowObject = window, fetcher = fetch, debounceMs = 250 }) {
    this.document = documentObject;
    this.window = windowObject;
    this.tooltip = tooltip;
    this.getSession = getSession;
    this.nameOf = nameOf;
    this.onSelection = onSelection;
    this.onRelease = onRelease;
    // Native browser fetch is receiver-sensitive in some web views. Calling a
    // detached function as this.fetcher would pass SelectionTranslator as this.
    this.fetcher = fetcher.bind(windowObject);
    this.debounceMs = debounceMs;
    this.cache = new BoundedCache(64);
    this.serial = 0;
    this.active = null;
    this.timer = null;
    this.controller = null;
    this.key = null;
    documentObject.addEventListener('selectionchange', () => this.changed());
    documentObject.addEventListener('pointerup', event => {
      if (event.target?.closest?.('.message-text')) this.changed();
    });
    documentObject.addEventListener('keyup', event => { if (event.shiftKey) this.changed(); });
    documentObject.addEventListener('keydown', event => { if (event.key === 'Escape') this.dismiss(); });
    documentObject.addEventListener('pointerdown', event => {
      if (!tooltip.contains(event.target)) this.dismiss();
    });
    tooltip.querySelector('[data-dismiss]').addEventListener('click', () => this.dismiss());
    windowObject.addEventListener('resize', () => this.position());
    documentObject.addEventListener('scroll', () => this.position(), true);
  }

  protectedCard() {
    if (!this.active || !this.key) return null;
    const selected = selectedMessage(this.window.getSelection());
    return selected?.block.closest('.message-card') ?? null;
  }

  changed() {
    clearTimeout(this.timer);
    const selected = selectedMessage(this.window.getSelection());
    if (!selected) {
      const released = Boolean(this.active);
      this.dismiss();
      this.active = null;
      if (released) this.onRelease?.();
      return;
    }
    const key = JSON.stringify([this.getSession(), selected.source_language, selected.target_language, selected.text]);
    if (key === this.key && this.active?.block === selected.block) { this.position(); return; }
    this.dismiss();
    this.active = selected;
    this.key = key;
    this.onSelection?.(selected.block);
    this.timer = setTimeout(() => this.lookup(key), this.debounceMs);
  }

  dismiss() {
    clearTimeout(this.timer);
    this.controller?.abort();
    this.controller = null;
    this.serial += 1;
    this.key = null;
    this.tooltip.hidden = true;
    this.active?.block.removeAttribute('aria-describedby');
  }

  reset() {
    this.dismiss();
    this.active = null;
  }

  show(text, status) {
    if (!this.active) return;
    const { source_language: source, target_language: target } = this.active;
    this.tooltip.querySelector('[data-languages]').textContent = `${this.nameOf(source)} → ${this.nameOf(target)}`;
    const content = this.tooltip.querySelector('[data-translation]');
    content.textContent = text;
    content.lang = target.toLowerCase();
    content.dir = ['AR', 'HE'].includes(target) ? 'rtl' : 'ltr';
    this.tooltip.dataset.status = status;
    this.tooltip.hidden = false;
    this.active.block.setAttribute('aria-describedby', this.tooltip.id);
    this.position();
  }

  position() {
    if (!this.active || this.tooltip.hidden || !this.active.block.isConnected) return;
    const range = this.active.range;
    const rect = range.getBoundingClientRect();
    const viewportWidth = this.window.innerWidth;
    const viewportHeight = this.window.innerHeight;
    const width = Math.min(370, viewportWidth - 24);
    this.tooltip.style.width = `${width}px`;
    const height = this.tooltip.getBoundingClientRect().height;
    const desiredX = ['AR', 'HE'].includes(this.active.source_language) ? rect.right - width : rect.left;
    const desiredY = rect.bottom + height + 10 < viewportHeight ? rect.bottom + 10 : rect.top - height - 10;
    this.tooltip.style.left = `${Math.max(12, Math.min(desiredX, viewportWidth - width - 12))}px`;
    this.tooltip.style.top = `${Math.max(12, Math.min(desiredY, viewportHeight - height - 12))}px`;
  }

  async request(path, body, signal) {
    const response = await this.fetcher(path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
      body: body ? JSON.stringify(body) : undefined, signal, cache: 'no-store', credentials: 'same-origin',
    });
    const result = await response.json();
    if (!response.ok || result.accepted === false || result.ok === false) throw new Error(result.detail || result.error || result.message || 'Selection translation is unavailable.');
    return result;
  }

  async lookup(key) {
    const selected = this.active;
    if (!selected || key !== this.key) return;
    const serial = ++this.serial;
    const session = this.getSession();
    const isCurrent = () => serial === this.serial && key === this.key && session === this.getSession();
    const cached = this.cache.get(key);
    if (cached !== undefined) { this.show(cached, 'complete'); return; }
    if (selected.text.length > 1200) { this.show('Select a shorter phrase (up to 1,200 characters).', 'error'); return; }
    this.controller = new AbortController();
    const signal = this.controller.signal;
    const selectionId = `selection-${serial}`;
    const timeout = setTimeout(() => { if (isCurrent()) this.controller?.abort(); }, 90000);
    this.show('Translating selection…', 'pending');
    try {
      let result = await this.request('/api/selection-translation', {
        text: selected.text, source_language: selected.source_language,
        target_language: selected.target_language, selection_id: selectionId,
      }, signal);
      const lookupId = result.lookup_id;
      while (isCurrent() && result.status !== 'complete' && result.status !== 'error') {
        if (!lookupId) throw new Error('The selection request was not queued.');
        this.show('Waiting for translation… Live audio continues.', 'pending');
        await abortableDelay(350, signal);
        result = await this.request(`/api/selection-translation/${encodeURIComponent(lookupId)}`, null, signal);
      }
      if (!isCurrent()) return;
      if (result.selection_id && result.selection_id !== selectionId) return;
      if ((result.source_language && result.source_language !== selected.source_language)
          || (result.target_language && result.target_language !== selected.target_language)) {
        throw new Error('Selection direction changed. Select the phrase again.');
      }
      if (result.status === 'error') throw new Error(result.error || result.message || 'Selection translation failed.');
      if (typeof result.translation !== 'string' || !result.translation.trim()) throw new Error('The model returned no selection translation.');
      this.cache.set(key, result.translation);
      this.show(result.translation, 'complete');
    } catch (error) {
      if (isCurrent()) this.show(error.name === 'AbortError' ? 'Translation timed out. Select the phrase again to retry.' : error.message, 'error');
    } finally {
      clearTimeout(timeout);
    }
  }
}
