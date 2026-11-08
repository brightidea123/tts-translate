import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AutoScrollController, BoundedCache, SelectionTranslator, selectedMessage, messageIsFinal } from '../web/ui-interactions.js';

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
