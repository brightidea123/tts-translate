import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AutoScrollController, BoundedCache, SelectionTranslator, selectedMessage, messageIsFinal, inputSelectionAvailable, selectInputDevice, refreshInputOptions, sourcePreviewSegments, renderSourcePreview, formatPhraseTiming, renderPhraseTiming } from '../web/ui-interactions.js';

test('phrase timing uses captured start time and updates live/final clip seconds', () => {
  const start = '2026-11-08T05:10:23.698Z';
  const live = formatPhraseTiming({ audio_started_at: start, audio_duration_seconds: 1.24 }, { timeZone: 'UTC' });
  assert.equal(live.text, '05:10:23 · 1.2 s');
  assert.match(live.title, /8 Nov 2026/);
  assert.match(live.title, /Speech clip duration: 1.2 seconds/);
  const final = formatPhraseTiming({ audio_started_at: start, audio_duration_seconds: 12.36, source_final: true }, { timeZone: 'UTC' });
  assert.equal(final.text, '05:10:23 · 12.4 s');
  assert.equal(formatPhraseTiming({ audio_started_at: start, audio_duration_seconds: 0 }, { timeZone: 'Europe/Istanbul' }).text, '08:10:23 · 0.0 s');
});

test('phrase timing omits missing or invalid metadata without inventing clip times', () => {
  for (const item of [{}, null, { audio_started_at: 'invalid', audio_duration_seconds: -1 }, { audio_started_at: '', audio_duration_seconds: NaN }]) {
    assert.deepEqual(formatPhraseTiming(item), { text: '', title: '' });
  }
  assert.equal(formatPhraseTiming({ audio_duration_seconds: 4.56 }).text, '4.6 s');
  assert.equal(formatPhraseTiming({ audio_started_at: '2026-11-08T05:10:23Z', audio_duration_seconds: '7' }, { timeZone: 'UTC' }).text, '05:10:23');
});

test('unchanged clip timing leaves its label untouched while source updates and live duration still advances', () => {
  let writes = 0, label = '';
  const element = { set textContent(value) { writes += 1; label = value; },
    setAttribute() { writes += 1; } };
  const item = { audio_started_at: '2026-11-08T05:10:23Z', audio_duration_seconds: 2.1, source: 'first reading' };
  renderPhraseTiming(element, item, { timeZone: 'UTC' });
  assert.equal(label, '05:10:23 · 2.1 s');
  const initialWrites = writes;
  renderPhraseTiming(element, { ...item, source: 'refined reading', source_inflight: true }, { timeZone: 'UTC' });
  assert.equal(writes, initialWrites);
  renderPhraseTiming(element, { ...item, audio_duration_seconds: 3.2 }, { timeZone: 'UTC' });
  assert.equal(label, '05:10:23 · 3.2 s');
  assert.equal(element.hidden, false);
  renderPhraseTiming(element, { ...item, audio_duration_seconds: 3.2 }, { timeZone: 'Europe/Istanbul' });
  assert.equal(label, '08:10:23 · 3.2 s');
  renderPhraseTiming(element, {}, { timeZone: 'UTC' });
  assert.equal(label, '');
  assert.equal(element.hidden, true);
});

function previewEnvironment() {
  const classes = new Set();
  let writes = 0;
  const content = { children: [], dataset: { sourceLanguage: 'HE', targetLanguage: 'AR' }, nodeType: 1,
    classList: { toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); }, contains(name) { return classes.has(name); } },
    replaceChildren(...nodes) { writes += 1; this.children = nodes; for (const node of nodes) node.parentElement = this; },
    append(node) { this.children.push(node); node.parentElement = this; },
    removeChild(node) { this.children = this.children.filter(child => child !== node); node.parentElement = null; },
    closest(selector) { return selector === '.message-text' ? this : 'card'; },
  };
  const documentObject = {
    createTextNode(text) { return { nodeType: 3, textContent: text }; },
    createElement() { return { nodeType: 1, textContent: '', closest(selector) { return this.parentElement.closest(selector); } }; },
  };
  return { content, documentObject, writes: () => writes,
    text: () => content.children.map(node => node.textContent).join('') };
}

test('source preview shows short placeholder and first words without stability gating', () => {
  const env = previewEnvironment();
  renderSourcePreview(env.content, { source_inflight: true }, '', env.documentObject);
  assert.equal(env.text(), '..');
  assert.equal(env.content.classList.contains('placeholder'), true);
  renderSourcePreview(env.content, {}, 'שלום', env.documentObject);
  assert.equal(env.text(), 'שלום');
  assert.equal(env.content.children[0].className, 'transcript-provisional');
  assert.equal(env.content.classList.contains('placeholder'), false);
  const writes = env.writes();
  renderSourcePreview(env.content, { source_inflight: true }, 'שלום', env.documentObject);
  renderSourcePreview(env.content, { source_inflight: false }, 'שלום', env.documentObject);
  assert.equal(env.writes(), writes);
  assert.equal(env.text(), 'שלום');
});

test('source preview revises uncertain words, preserves mixed RTL spacing and clears styling at final', () => {
  const env = previewEnvironment();
  const text = 'שלום  world <script> & مرحبا';
  renderSourcePreview(env.content, { source_stable: 'שלום' }, text, env.documentObject);
  assert.equal(env.text(), text);
  assert.equal(env.content.children[0].textContent, 'שלום');
  assert.equal(env.content.children[1].textContent, '  world <script> & مرحبا');
  assert.equal(env.content.children[1].className, 'transcript-provisional');
  const selection = { isCollapsed: false, rangeCount: 1, anchorNode: env.content.children[0], focusNode: env.content.children[1],
    toString: () => text, getRangeAt: () => ({}) };
  assert.equal(selectedMessage(selection).text, text);
  renderSourcePreview(env.content, { source_stable: 'שלום' }, 'שלום חברים', env.documentObject);
  assert.equal(env.text(), 'שלום חברים');
  renderSourcePreview(env.content, { source_final: true, source_stable: 'שלום' }, 'שלום חברים', env.documentObject);
  assert.equal(env.content.children.length, 1);
  assert.equal(env.content.children[0].nodeType, 3);
  assert.equal(env.text(), 'שלום חברים');
});

test('stale stable-prefix metadata never hides a corrected provisional reading', () => {
  assert.deepEqual(sourcePreviewSegments({ source_stable: 'old prefix' }, 'new reading'),
    { text: 'new reading', stable: '', provisional: 'new reading', placeholder: false });
  assert.equal(sourcePreviewSegments({}, '..').placeholder, true);
});

test('steady source node survives provisional-tail edits and authoritative corrections replace safely', () => {
  const env = previewEnvironment();
  renderSourcePreview(env.content, { source_stable: 'שלום' }, 'שלום עולם', env.documentObject);
  const steady = env.content.children[0];
  const provisional = env.content.children[1];
  const writes = env.writes();
  renderSourcePreview(env.content, { source_stable: 'שלום' }, 'שלום חברים חדשים', env.documentObject);
  assert.equal(env.content.children[0], steady);
  assert.equal(env.content.children[1], provisional);
  assert.equal(steady.textContent, 'שלום');
  assert.equal(env.text(), 'שלום חברים חדשים');
  assert.equal(env.writes(), writes);
  renderSourcePreview(env.content, { source_stable: 'שלום' }, 'שלום', env.documentObject);
  assert.equal(env.content.children[0], steady);
  assert.equal(env.content.children.length, 1);
  renderSourcePreview(env.content, { source_stable: 'שלום' }, 'שלום שוב', env.documentObject);
  assert.equal(env.content.children[0], steady);
  assert.equal(env.text(), 'שלום שוב');
  renderSourcePreview(env.content, { source_final: true }, 'ערב טוב', env.documentObject);
  assert.equal(env.text(), 'ערב טוב');
  assert.equal(env.content.children.length, 1);
  assert.equal(env.content.children[0].nodeType, 3);
  assert.notEqual(env.content.children[0], steady);
});

class DeviceOption {
  constructor(text, value) { this.text = text; this.value = value; this.disabled = false; }
}
function deviceSelect(value) {
  return { value, options: [], add(option) { this.options.push(option); },
    replaceChildren(...options) { this.options = options; this.value = options[0]?.value; } };
}

test('audio input names survive renumbering and old numeric selections require reselection', () => {
  const select = deviceSelect('LitLink');
  refreshInputOptions(select, [{ name: 'LitLink', index: 7, max_input_channels: 2 }], DeviceOption);
  assert.equal(select.value, 'LitLink');
  assert.equal(inputSelectionAvailable(select), true);
  refreshInputOptions(select, [{ name: 'LitLink', index: 0, max_input_channels: 2 }], DeviceOption);
  assert.equal(select.value, 'LitLink');
  assert.equal(inputSelectionAvailable(select), true);
  select.value = '0';
  refreshInputOptions(select, [{ name: 'Different source', index: 0, max_input_channels: 1 }], DeviceOption);
  assert.equal(select.value, '0');
  assert.equal(inputSelectionAvailable(select), false);
});

test('missing or ambiguous input never silently selects a microphone', () => {
  const select = deviceSelect('LitLink');
  refreshInputOptions(select, [{ name: 'Mic', max_input_channels: 1 }, { name: 'Speakers', max_input_channels: 0 }], DeviceOption);
  assert.equal(select.value, 'LitLink');
  assert.equal(inputSelectionAvailable(select), false);
  assert.equal(select.options.some(option => option.value === 'Speakers'), false);
  selectInputDevice(select, 'Mic', DeviceOption);
  assert.equal(inputSelectionAvailable(select), true);
  refreshInputOptions(select, [{ name: 'Mic', max_input_channels: 1, ambiguous: true }], DeviceOption);
  assert.equal(inputSelectionAvailable(select), false);
  selectInputDevice(select, '', DeviceOption);
  assert.equal(inputSelectionAvailable(select), true);
});

class Element {
  constructor() { this.listeners = {}; this.attributes = {}; this.dataset = {}; this.style = {}; this.isConnected = true; this.nodeType = 1; this.checked = true; this.hidden = true; this.scrollTop = 300; this.scrollHeight = 700; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  emit(name, event = {}) { for (const fn of this.listeners[name] || []) fn(event); }
  setAttribute(k, v) { this.attributes[k] = v; }
  removeAttribute(k) { delete this.attributes[k]; }
  contains(node) { return node === this; }
  getBoundingClientRect() { return { top: 10, bottom: 210, left: 15, right: 215, height: 200 }; }
}

function environment(fetcher) {
  const documentObject = new Element(), windowObject = new Element(), block = new Element(), tooltip = new Element();
  const children = Object.fromEntries(['data-dismiss', 'data-languages', 'data-translation'].map(key => [key, new Element()]));
  tooltip.id = 'tip'; tooltip.querySelector = selector => children[selector.slice(1, -1)];
  block.dataset = { sourceLanguage: 'HE', targetLanguage: 'AR' };
  block.classList = { contains: () => false }; block.closest = selector => selector === '.message-text' ? block : 'card';
  const selection = { isCollapsed: false, rangeCount: 1, anchorNode: block, focusNode: block,
    toString: () => 'שלום עולם', getRangeAt: () => ({ getBoundingClientRect: () => ({ top: 20, bottom: 40, left: 30, right: 140 }) }) };
  windowObject.innerWidth = 800; windowObject.innerHeight = 600; windowObject.getSelection = () => selection;
  const session = { value: 'run-one' };
  const translator = new SelectionTranslator({ tooltip, getSession: () => session.value, nameOf: code => code,
    documentObject, windowObject, fetcher, debounceMs: 1 });
  const selected = selectedMessage(selection);
  const key = JSON.stringify([session.value, selected.source_language, selected.target_language, selected.text]);
  translator.active = selected; translator.key = key;
  return { translator, tooltip, children, session, key, selection, block };
}

test('unfinished messages stay active until both source and translation are final', () => {
  assert.equal(messageIsFinal({source_final: true, translation_final: false}), false);
  assert.equal(messageIsFinal({source_final: false, translation_final: true}), false);
  assert.equal(messageIsFinal({source_final: true, translation_final: true}), true);
});

test('upward scroll turns follow off until explicitly checked again', () => {
  const scroller = new Element(), checkbox = new Element();
  const control = new AutoScrollController(scroller, checkbox);
  scroller.emit('wheel', {deltaY: -1}); assert.equal(checkbox.checked, false);
  const snapshot = control.beforeRender({children: []}); scroller.scrollHeight = 950;
  control.afterRender(snapshot); assert.equal(scroller.scrollTop, 300);
  scroller.scrollTop = 400; scroller.emit('scroll'); assert.equal(checkbox.checked, false);
  checkbox.checked = true; checkbox.emit('change'); assert.equal(scroller.scrollTop, 950);
  scroller.scrollTop = 600; scroller.emit('scroll'); assert.equal(checkbox.checked, false);
  checkbox.checked = true; scroller.emit('keydown', {key: 'PageUp'}); assert.equal(checkbox.checked, false);
  checkbox.checked = true; scroller.emit('touchstart', {touches: [{clientY: 50}]});
  scroller.emit('touchmove', {touches: [{clientY: 80}]}); assert.equal(checkbox.checked, false);
});

test('paused follow keeps a visible anchor when bounded history removes old cards', () => {
  const scroller = new Element(), checkbox = new Element(); checkbox.checked = false;
  const card = new Element(); let top = 30;
  card.getBoundingClientRect = () => ({top, bottom: top + 90});
  const control = new AutoScrollController(scroller, checkbox);
  const before = control.beforeRender({children: [card]}); top = -20;
  control.afterRender(before); assert.equal(scroller.scrollTop, 250);
  assert.equal(checkbox.checked, false);
});

test('selection uses the selected block direction and rejects mixed blocks/placeholders', () => {
  const e = environment();
  assert.equal(selectedMessage(e.selection).text, 'שלום עולם');
  assert.equal(selectedMessage(e.selection).target_language, 'AR');
  e.selection.focusNode = {nodeType: 1, closest: () => new Element()};
  assert.equal(selectedMessage(e.selection), null);
  e.selection.focusNode = e.block; e.block.classList.contains = () => true;
  assert.equal(selectedMessage(e.selection), null);
});

test('tooltip request is isolated, RTL, text-only and cached without speech/reply actions', async () => {
  const calls = [];
  const e = environment(async (url, options) => {
    calls.push([url, JSON.parse(options.body)]);
    return {ok: true, json: async () => ({accepted: true, status: 'complete', selection_id: JSON.parse(options.body).selection_id, translation: 'مرحبا <b>بكم</b>'})};
  });
  await e.translator.lookup(e.key);
  assert.equal(calls[0][0], '/api/selection-translation');
  assert.equal(calls[0][1].source_language, 'HE'); assert.equal(calls[0][1].target_language, 'AR');
  assert.equal(e.children['data-translation'].dir, 'rtl');
  assert.equal(e.children['data-translation'].textContent, 'مرحبا <b>بكم</b>');
  await e.translator.lookup(e.key); assert.equal(calls.length, 1);
  e.translator.dismiss(); assert.equal(e.tooltip.hidden, true);
  assert.equal(e.translator.protectedCard(), null);
});

test('dismissed or older-session tooltip results cannot overwrite a new selection', async () => {
  let finish;
  const e = environment(() => new Promise(resolve => { finish = resolve; }));
  const pending = e.translator.lookup(e.key);
  e.translator.reset(); e.session.value = 'run-two';
  finish({ok: true, json: async () => ({status: 'complete', translation: 'stale answer'})});
  await pending; assert.equal(e.tooltip.hidden, true); assert.equal(e.translator.cache.values.size, 0);
});

test('tooltip LRU stays bounded and refreshes recent phrases', () => {
  const cache = new BoundedCache(2); cache.set('one', 1); cache.set('two', 2); cache.get('one'); cache.set('three', 3);
  assert.equal(cache.get('two'), undefined); assert.equal(cache.get('one'), 1); assert.equal(cache.values.size, 2);
});

test('pending selection polls its own identity before displaying completion', async () => {
  let config; const urls = [];
  const e = environment(async (url, options) => {
    urls.push(url);
    if (options.method === 'POST') {
      config = JSON.parse(options.body);
      return {ok: true, json: async () => ({accepted: true, status: 'pending', lookup_id: 'a'.repeat(32), ...config})};
    }
    return {ok: true, json: async () => ({status: 'complete', translation: 'مرحبا', ...config})};
  });
  await e.translator.lookup(e.key);
  assert.equal(urls.length, 2); assert.equal(urls[1], `/api/selection-translation/${'a'.repeat(32)}`);
  assert.equal(e.children['data-translation'].textContent, 'مرحبا');
});

test('browser fetch retains its Window receiver for selection POST and result polling', async () => {
  let windowReceiver, config;
  const methods = [];
  const e = environment(function (url, options) {
    assert.equal(this, windowReceiver, 'window.fetch requires its Window receiver');
    methods.push(options.method);
    if (options.method === 'POST') {
      config = JSON.parse(options.body);
      return Promise.resolve({ok: true, json: async () => ({status: 'pending', lookup_id: 'b'.repeat(32), ...config})});
    }
    return Promise.resolve({ok: true, json: async () => ({status: 'complete', translation: 'مرحباً بالعالم', ...config})});
  });
  windowReceiver = e.translator.window;
  await e.translator.lookup(e.key);
  assert.equal(e.tooltip.dataset.status, 'complete');
  assert.equal(e.children['data-translation'].textContent, 'مرحباً بالعالم');
  assert.deepEqual(methods, ['POST', 'GET']);
});

test('clicking another control does not reopen a dismissed selection or reset its checkbox', async () => {
  const e = environment();
  e.translator.dismiss();
  e.translator.document.emit('pointerup', {target: {closest: () => null}});
  assert.equal(e.translator.key, null);
  assert.equal(e.tooltip.hidden, true);
  assert.equal(e.translator.timer, null);
});
