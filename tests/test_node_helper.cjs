// SPDX-License-Identifier: Apache-2.0
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  const shells = [];
  class PythonShell {
    constructor(script, options) {
      this.script = script;
      this.options = options;
      this.handlers = {};
      this.commands = [];
      this.childProcess = {
        killed: false,
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
  }
  const context = {
    module: { exports: {} },
    console: { log() {}, error() {} },
    require: name => ({ node_helper: { create: value => value }, 'python-shell': { PythonShell }, 'signal-exit': () => {} })[name],
  };
  vm.runInNewContext(fs.readFileSync('node_helper.js', 'utf8'), context);
  const helper = context.module.exports;
  helper.name = 'MMM-Face-Reco-DNN';
  helper.sendSocketNotification = () => {};
  helper.config = { external_trigger_notification: '', cameraLores: true, cameraLoresFormat: 'RGB888', cameraFrameRate: 15, method: 'yunet', pythonPath: '/venv/python' };
  return { helper, shells };
}

test('forwards supported acceleration controls without losing Python path', () => {
  const { helper, shells } = setup();
  helper.python_start();
  assert.equal(shells[0].options.pythonPath, '/venv/python');
  assert.ok(shells[0].options.args.includes('--cameraLores=True'));
  assert.ok(shells[0].options.args.includes('--cameraLoresFormat=RGB888'));
  assert.ok(shells[0].options.args.includes('--cameraFrameRate=15'));
  assert.ok(shells[0].options.args.includes('--method=yunet'));
});
test('destroy is idempotent and releases the Python process with SIGTERM', () => {
  const { helper, shells } = setup();
  helper.python_start();
  helper.destroy();
  helper.destroy();
  assert.equal(shells[0].signal, 'SIGTERM');
  assert.equal(helper.pyshell, null);
});
test('stop and notification before configuration do not dereference missing worker', () => {
  const { helper } = setup();
  helper.config = null;
  helper.socketNotificationReceived('other', false);
  helper.stop();
  helper.send_python_cmd('stop');
});
