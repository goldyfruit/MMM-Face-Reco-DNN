// SPDX-License-Identifier: Apache-2.0
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { PythonShell: RealPythonShell } = require('python-shell');

function fakeClock() {
  const timers = new Map();
  let nextId = 0;
  let now = 0;
  return {
    timers,
    now: () => now,
    setTimeout(fn, delay) {
      const id = ++nextId;
      timers.set(id, { fn, at: now + delay, delay });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    advance(milliseconds) {
      now += milliseconds;
      for (const [id, task] of [...timers]) {
        if (timers.has(id) && task.at <= now) {
          timers.delete(id);
          task.fn();
        }
      }
    },
  };
}

function setup({ failedStarts = 0 } = {}) {
  const shells = [];
  const notifications = [];
  const logs = [];
  const exitCallbacks = [];
  const clock = fakeClock();
  let constructionAttempts = 0;
  class PythonShell {
    constructor(script, options) {
      constructionAttempts++;
      if (failedStarts-- > 0) throw Error('private-gallery-label must not be logged');
      this.script = script;
      this.options = options;
      this.handlers = {};
      this.childHandlers = {};
      this.commands = [];
      this.childProcess = {
        killed: false,
        exitCode: null,
        signalCode: null,
        once: (event, fn) => {
          this.childHandlers[event] = fn;
        },
        kill: signal => {
          this.signal = signal;
          this.childProcess.killed = true;
        },
      };
      shells.push(this);
    }
    on(event, fn) {
      this.handlers[event] = fn;
    }
    send(command) {
      this.commands.push(command);
    }
    message(value) {
      this.handlers.message(value);
    }
    close(code = 0) {
      this.childProcess.exitCode = code;
      this.handlers.close();
      this.childHandlers.close();
    }
  }
  const context = {
    module: { exports: {} },
    Date: { now: clock.now },
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    console: { log: text => logs.push(text), error: text => logs.push(text) },
    require: name => ({ node_helper: { create: value => value }, 'python-shell': { PythonShell }, 'signal-exit': fn => exitCallbacks.push(fn) })[name],
  };
  vm.runInNewContext(fs.readFileSync('node_helper.js', 'utf8'), context);
  const helper = context.module.exports;
  helper.name = 'MMM-Face-Reco-DNN';
  helper.sendSocketNotification = (name, payload) => notifications.push({ name, payload });
  helper.config = {
    external_trigger_notification: '',
    cameraLores: true,
    cameraLoresFormat: 'RGB888',
    cameraFrameRate: 15,
    method: 'yunet',
    pythonPath: '/venv/python',
  };
  return { helper, shells, notifications, logs, clock, exitCallbacks, constructionAttempts: () => constructionAttempts };
}

function setupFrontend() {
  const clock = fakeClock();
  const notifications = [];
  const hidden = [];
  const shown = [];
  let module;
  const context = {
    Module: {
      register: (_, value) => {
        module = value;
      },
    },
    Log: { log() {} },
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    document: {
      createElement: tag => ({
        tag,
        children: [],
        classList: { add() {} },
        appendChild(child) {
          this.children.push(child);
        },
      }),
    },
  };
  vm.runInNewContext(fs.readFileSync('MMM-Face-Reco-DNN.js', 'utf8'), context);
  module.config = { ...module.defaults, classes_noface: ['noface', 'default', 'always'], classes_known: ['known'], classes_unknown: ['unknown'], debug: 0 };
  module.users = ['private-person'];
  module.userClasses = { 'private-person': ['known', 'private-person'] };
  module.image = 'old-face-image';
  module.timouts = {
    'private-person': clock.setTimeout(() => {
      throw Error('stale logout timer ran');
    }, 15000),
  };
  module.hide_modules = (classes, except) => hidden.push([classes, except]);
  module.show_modules = (classes, except) => shown.push([classes, except]);
  module.sendNotification = (name, payload) => notifications.push([name, payload]);
  module.updateDom = () => {};
  return { module, clock, notifications, hidden, shown };
}

test('forwards supported camera controls and preserves the Python path', () => {
  const { helper, shells } = setup();
  helper.python_start();
  const options = shells[0].options;
  assert.equal(options.pythonPath, '/venv/python');
  for (const argument of [
    '--cameraLores=True',
    '--cameraLoresFormat=RGB888',
    '--cameraFrameRate=15',
    '--method=yunet',
    '--opencvThreads=0',
    '--cameraAutofocus=default',
    '--cameraExposureMode=normal',
  ]) {
    assert.ok(options.args.includes(argument));
  }
  assert.ok(!options.args.some(argument => argument.startsWith('--cameraLensPosition=')));
});

test('forwards explicit thread, focus and exposure settings', () => {
  for (const mode of ['manual', 'face']) {
    const { helper, shells } = setup();
    Object.assign(helper.config, { opencvThreads: 2, cameraAutofocus: mode, cameraLensPosition: 1.0, cameraExposureMode: 'short' });
    helper.python_start();
    for (const argument of ['--opencvThreads=2', '--cameraAutofocus=' + mode, '--cameraLensPosition=1', '--cameraExposureMode=short']) {
      assert.ok(shells[0].options.args.includes(argument));
    }
  }
});

test('worker errors clear availability immediately but never overlap camera owners', () => {
  const { helper, shells, notifications, clock, logs } = setup();
  helper.python_start();
  const first = shells[0];
  first.message({ ready: true });
  first.message({ login: { names: ['private-person'] } });
  first.handlers.pythonError(Error('private-person error'));
  assert.equal(first.signal, 'SIGTERM');
  assert.equal(notifications.at(-1).name, 'worker_state');
  assert.equal(notifications.at(-1).payload.available, false);
  assert.equal(clock.timers.size, 0);
  assert.equal(shells.length, 1);
  const count = notifications.length;
  first.message({ login: { names: ['stale-person'] } });
  assert.equal(notifications.length, count);
  first.close(1);
  assert.equal(clock.timers.size, 1);
  assert.equal([...clock.timers.values()][0].delay, 1000);
  clock.advance(1000);
  assert.equal(shells.length, 2);
  assert.ok(logs.every(line => !line.includes('private-person') && !line.includes('stale-person')));
});

test('duplicate failure and close events schedule a single retry', () => {
  const { helper, shells, clock } = setup();
  helper.python_start();
  shells[0].handlers.error(Error('spawn failure'));
  shells[0].handlers.pythonError(Error('duplicate'));
  shells[0].close(1);
  shells[0].close(1);
  assert.equal(clock.timers.size, 1);
  clock.advance(1000);
  assert.equal(shells.length, 2);
});

test('spawn failures recover from child close even without PythonShell close', () => {
  const { helper, shells, clock } = setup();
  helper.python_start();
  shells[0].handlers.error(Error('spawn failure'));
  shells[0].childHandlers.close();
  assert.equal(helper.pyshell, null);
  clock.advance(1000);
  assert.equal(shells.length, 2);
});

test('early failures exhaust a bounded exponential retry budget', () => {
  const { helper, shells, clock, exitCallbacks } = setup();
  helper.python_start();
  for (const delay of [1000, 2000, 4000]) {
    shells.at(-1).close(1);
    assert.equal([...clock.timers.values()][0].delay, delay);
    clock.advance(delay);
  }
  shells.at(-1).close(1);
  assert.equal(shells.length, 4);
  assert.equal(clock.timers.size, 0);
  assert.equal(exitCallbacks.length, 1);
});

test('long-running worker receives a fresh recovery budget', () => {
  const { helper, shells, clock } = setup();
  helper.config.workerRestartLimit = 1;
  helper.python_start();
  shells[0].close(1);
  clock.advance(1000);
  clock.advance(60000);
  shells[1].close(1);
  assert.equal([...clock.timers.values()][0].delay, 1000);
});

test('synchronous startup exceptions use the same budget without logging private arguments', () => {
  const state = setup({ failedStarts: Infinity });
  state.helper.python_start();
  for (const delay of [1000, 2000, 4000]) state.clock.advance(delay);
  assert.equal(state.constructionAttempts(), 4);
  assert.equal(state.clock.timers.size, 0);
  assert.ok(state.logs.every(line => !line.includes('private-gallery-label')));
});

test('retry policy supports disabling recovery and validates unsafe values', () => {
  const disabled = setup();
  disabled.helper.config.workerRestartLimit = 0;
  disabled.helper.python_start();
  disabled.shells[0].close(1);
  assert.equal(disabled.clock.timers.size, 0);
  const invalid = setup();
  invalid.helper.config.workerRestartDelay = -1;
  invalid.helper.config.workerRestartLimit = 1000;
  invalid.helper.python_start();
  invalid.shells[0].close(1);
  assert.equal([...invalid.clock.timers.values()][0].delay, 1000);
});

test('explicit stop cancels retries and destroy waits for child closure idempotently', () => {
  const { helper, shells, clock } = setup();
  helper.python_start();
  helper.destroy();
  helper.destroy();
  assert.equal(shells[0].signal, 'SIGTERM');
  assert.equal(helper.pyshell, shells[0]);
  shells[0].close();
  assert.equal(helper.pyshell, null);
  assert.equal(clock.timers.size, 0);
  helper.socketNotificationReceived('CONFIG', helper.config);
  shells.at(-1).close(1);
  assert.equal(clock.timers.size, 1);
  helper.stop();
  clock.advance(10000);
  assert.equal(shells.length, 2);
});

test('reconfiguration during shutdown starts only after the old child closes', () => {
  const { helper, shells, notifications } = setup();
  helper.python_start();
  const first = shells[0];
  helper.stop();
  helper.socketNotificationReceived('CONFIG', helper.config);
  assert.equal(shells.length, 1);
  const count = notifications.length;
  first.message({ ready: true });
  first.message({ login: { names: ['stale-session'] } });
  first.message({ camera_image: { image: 'stale-image' } });
  assert.equal(notifications.length, count);
  first.close();
  assert.equal(shells.length, 2);
  first.close();
  assert.equal(helper.pyshell, shells[1]);
});

test('external pause survives worker failure and pending retry cancellation', () => {
  const { helper, shells, clock, notifications } = setup();
  helper.config.external_trigger_notification = 'PIR';
  helper.python_start();
  helper.socketNotificationReceived('PIR', false);
  assert.deepEqual(shells[0].commands, ['stop']);
  assert.equal(notifications.at(-1).payload.available, false);
  const count = notifications.length;
  shells[0].message({ ready: true });
  shells[0].message({ login: { names: ['paused-session'] } });
  shells[0].message({ camera_image: { image: 'paused-image' } });
  assert.equal(notifications.length, count);
  shells[0].close(1);
  assert.equal(clock.timers.size, 0);
  helper.socketNotificationReceived('PIR', true);
  assert.equal(clock.timers.size, 1);
  helper.socketNotificationReceived('PIR', false);
  assert.equal(clock.timers.size, 0);
  helper.socketNotificationReceived('PIR', true);
  clock.advance(1000);
  assert.equal(shells.length, 2);
});

test('the installed PythonShell formatter sends exact plain commands while parsing JSON output', () => {
  const { helper, shells } = setup();
  helper.python_start();
  const options = shells[0].options;
  const writes = [];
  const nativeSender = {
    mode: options.mode,
    formatter: RealPythonShell.format[options.formatter],
    stdin: { write: line => writes.push(line) },
  };
  RealPythonShell.prototype.send.call(nativeSender, 'stop');
  RealPythonShell.prototype.send.call(nativeSender, 'start');
  assert.deepEqual(writes, ['stop\n', 'start\n']);
  assert.deepEqual(RealPythonShell.parse[options.mode]('{"ready":true}'), { ready: true });
});

test('reconfiguration cannot start a replacement while the external trigger is paused', () => {
  const { helper, shells, clock } = setup();
  helper.config.external_trigger_notification = 'PIR';
  helper.python_start();
  helper.socketNotificationReceived('PIR', false);
  shells[0].close(1);
  helper.socketNotificationReceived('CONFIG', helper.config);
  assert.equal(shells.length, 1);
  assert.equal(clock.timers.size, 0);
});

test('termination errors fail closed without crashing other mirror modules or starting another owner', () => {
  const { helper, shells, clock, notifications } = setup();
  helper.python_start();
  shells[0].childProcess.kill = () => {
    throw Error('private OS detail');
  };
  assert.doesNotThrow(() => shells[0].handlers.error(Error('worker failed')));
  assert.equal(notifications.at(-1).payload.available, false);
  assert.equal(clock.timers.size, 0);
  assert.equal(helper.pyshell, shells[0]);
});

test('late events from an old process cannot change a recovered worker', () => {
  const { helper, shells, notifications, clock } = setup();
  helper.python_start();
  const first = shells[0];
  first.close(1);
  clock.advance(1000);
  shells[1].message({ ready: true });
  const count = notifications.length;
  first.message({ login: { names: ['old-session'] } });
  first.handlers.error(Error('late failure'));
  first.close(1);
  assert.equal(notifications.length, count);
  assert.equal(helper.pyshell, shells[1]);
});

test('malformed worker events and stop before configuration are harmless', () => {
  const { helper, shells, notifications } = setup();
  helper.python_start();
  for (const message of [null, {}, { camera_image: null }, { camera_image: { image: 3 } }, { login: { names: null } }]) shells[0].message(message);
  assert.equal(notifications.length, 0);
  helper.config = null;
  assert.doesNotThrow(() => helper.socketNotificationReceived('CONFIG', null));
  helper.socketNotificationReceived('other', false);
  helper.stop();
  helper.send_python_cmd('stop');
});

test('worker unavailability immediately clears frontend sessions, timers and preview', () => {
  const { module, clock, notifications, hidden, shown } = setupFrontend();
  module.socketNotificationReceived('worker_state', { available: false });
  assert.equal(module.users.length, 0);
  assert.equal(Object.keys(module.userClasses).length, 0);
  assert.equal(Object.keys(module.timouts).length, 0);
  assert.equal(module.image, '');
  assert.equal(clock.timers.size, 0);
  assert.equal(hidden.length, 1);
  assert.equal(shown.length, 1);
  assert.ok(notifications.some(([name]) => name === 'USERS_LOGOUT_MODULES'));
  clock.advance(60000);
  module.socketNotificationReceived('worker_state', { available: true });
  assert.equal(module.users.length, 0);
  assert.equal(module.workerAvailable, true);
});

test('unavailable frontend shows status rather than stale image', () => {
  const { module } = setupFrontend();
  module.socketNotificationReceived('worker_state', { available: false });
  const view = module.getDom();
  assert.equal(view.children[0].tag, 'span');
  assert.equal(view.children[0].textContent, 'Face recognition unavailable');
  assert.equal(
    view.children.some(child => child.tag === 'img'),
    false,
  );
});

test('frontend ignores malformed and unrelated session payloads', () => {
  const { module } = setupFrontend();
  module.socketNotificationReceived('user', null);
  module.socketNotificationReceived('user', { action: 'login', users: null });
  module.socketNotificationReceived('other', { action: 'login', users: ['intruder'] });
  assert.equal(module.users.length, 1);
  assert.equal(module.users[0], 'private-person');
});

test('each disappearing user is logged out after the configured delay', () => {
  const { module, clock, notifications } = setupFrontend();
  module.reset_sessions();
  notifications.length = 0;
  module.users = ['first', 'second'];
  module.userClasses = { first: ['known', 'first'], second: ['known', 'second'] };
  module.socketNotificationReceived('user', { action: 'logout', users: ['first', 'second'] });
  assert.equal(module.users.length, 2);
  assert.equal(clock.timers.size, 2);
  assert.equal(notifications.length, 0);
  clock.advance(module.config.logoutDelay);
  assert.equal(module.users.length, 0);
  assert.equal(Object.keys(module.timouts).length, 0);
  assert.deepEqual(
    notifications.filter(([name]) => name === 'USERS_LOGOUT_MODULES').map(([, user]) => user),
    ['first', 'second'],
  );
  assert.deepEqual(
    notifications.filter(([name]) => name === 'USERS_LOGOUT').map(([, users]) => users[0]),
    ['first', 'second'],
  );
});

test('duplicate logout notifications cannot leave an uncancellable timer after the face returns', () => {
  const { module, clock } = setupFrontend();
  module.reset_sessions();
  module.users = ['subject'];
  module.userClasses = { subject: ['known', 'subject'] };
  module.socketNotificationReceived('user', { action: 'logout', users: ['subject'] });
  module.socketNotificationReceived('user', { action: 'logout', users: ['subject'] });
  assert.equal(clock.timers.size, 1);
  module.socketNotificationReceived('user', { action: 'login', users: ['subject'] });
  assert.equal(clock.timers.size, 0);
  clock.advance(module.config.logoutDelay);
  assert.equal(module.users[0], 'subject');
});

test('fresh presence is bounded and forwarded independently from cached login state', () => {
  const { helper, shells, notifications } = setup();
  helper.python_start();
  const value = { version: 1, epoch: 'a'.repeat(32), sequence: 1, observed_at: 1000, available: true, labels: ['subject'] };
  shells[0].message({ presence: value });
  assert.equal(notifications.at(-1).name, 'presence');
  assert.deepEqual(JSON.parse(JSON.stringify(notifications.at(-1).payload)), value);
  const count = notifications.length;
  for (const patch of [
    { sequence: true },
    { sequence: -1 },
    { labels: ['x'.repeat(101)] },
    { labels: ['hidden\u202etext'] },
    { labels: ['bad\ud800'] },
    { available: false },
    { image: 'private-pixels' },
    { epoch: 'bad' },
    { epoch: undefined },
  ])
    shells[0].message({ presence: { ...value, ...patch } });
  assert.equal(notifications.length, count);
  shells[0].message({ presence: { ...value, labels: ['subject', 'subject'] } });
  assert.deepEqual(Array.from(notifications.at(-1).payload.labels), ['subject', 'subject']);
  assert.equal(notifications.at(-1).payload.labels.length, 2);
  const front = setupFrontend();
  front.module.socketNotificationReceived('presence', value);
  assert.equal(front.notifications.at(-1)[0], 'FACE_RECOGNITION_PRESENCE');
});

test('explicit Christmas keepsake statuses are bounded and separate from face preview', () => {
  const { helper, shells, notifications, logs } = setup();
  helper.config.christmasPhotoDirectory = '/private/keepsakes';
  helper.config.christmasPhotoTimezone = 'Europe/Paris';
  helper.python_start();
  assert.ok(shells[0].options.args.includes('--christmasPhotoDirectory=/private/keepsakes'));
  assert.ok(shells[0].options.args.includes('--christmasPhotoTimezone=Europe/Paris'));
  const image = Buffer.from([255, 216, 255, 217]).toString('base64');
  const value = { version: 1, revision: 7, capture_revision: 6, status: 'preview', image };
  shells[0].message({ christmas_photo: value });
  assert.equal(notifications.at(-1).name, 'christmas_photo');
  assert.deepEqual(JSON.parse(JSON.stringify(notifications.at(-1).payload)), value);
  assert.ok(!logs.join(' ').includes(image));
  const before = notifications.length;
  for (const patch of [
    { version: true },
    { revision: -1 },
    { capture_revision: 8 },
    { capture_revision: true },
    { status: 'upload' },
    { image: 'https://example.invalid/' },
    { image: 'A'.repeat(700000) },
    { image, filename: '/private/file.jpg' },
    { status: 'saved', image },
    { status: 'countdown', image, seconds: 4 },
  ]) {
    shells[0].message({ christmas_photo: { ...value, ...patch } });
  }
  assert.equal(notifications.length, before);
  shells[0].message({ christmas_photo: { version: 1, revision: 8, capture_revision: 8, status: 'countdown', seconds: 3 } });
  assert.equal(notifications.at(-1).payload.status, 'countdown');
  shells[0].message({ christmas_photo: { version: 1, revision: 9, capture_revision: 8, status: 'saved' } });
  assert.equal(notifications.at(-1).payload.status, 'saved');
  const front = setupFrontend();
  front.module.socketNotificationReceived('christmas_photo', value);
  assert.equal(front.notifications.at(-1)[0], 'CHRISTMAS_PHOTO_STATUS');
  assert.deepEqual(front.notifications.at(-1)[1], value);
});

test('failed or paused camera worker cannot forward late keepsake pixels', () => {
  const { helper, shells, notifications } = setup();
  helper.python_start();
  const old = shells[0];
  old.handlers.error(Error('failure'));
  const before = notifications.length;
  old.message({ christmas_photo: { version: 1, revision: 4, status: 'preview', image: '/9j/2Q==' } });
  assert.equal(notifications.length, before);
});
