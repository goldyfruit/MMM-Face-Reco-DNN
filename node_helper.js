/* Magic Mirror
 * Module: MMM-Face-Reco-DNN
 *
 * By Thierry Nischelwitzer http://nischi.ch
 * MIT Licensed.
 */

/*global require, module*/

'use strict';

const NodeHelper = require('node_helper');
const { PythonShell } = require('python-shell');
const onExit = require('signal-exit');

// Fixed metadata only; duplicate labels preserve the actual face count.
function validPresence(value) {
  return (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value).length === 6 &&
    ['version', 'epoch', 'sequence', 'observed_at', 'available', 'labels'].every(key => Object.prototype.hasOwnProperty.call(value, key)) &&
    value.version === 1 &&
    typeof value.epoch === 'string' &&
    /^[a-f0-9]{32}$/.test(value.epoch) &&
    Number.isSafeInteger(value.sequence) &&
    value.sequence > 0 &&
    Number.isFinite(value.observed_at) &&
    value.observed_at >= 0 &&
    typeof value.available === 'boolean' &&
    Array.isArray(value.labels) &&
    value.labels.length <= 8 &&
    value.labels.every(label => typeof label === 'string' && label.length > 0 && label.length <= 100 && !/[\p{Cc}\p{Cf}\p{Cs}]/u.test(label)) &&
    (value.available || value.labels.length === 0)
  );
}

// Explicit keepsake output is separate from face enrollment and ordinary preview.
function validChristmasPhoto(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.version !== 1 || !Number.isSafeInteger(value.revision) || value.revision < 0) return false;
  const status = value.status;
  if (!['countdown', 'preview', 'saved', 'discarded', 'unavailable'].includes(status)) return false;
  const capture = Object.prototype.hasOwnProperty.call(value, 'capture_revision');
  if (capture && (!Number.isSafeInteger(value.capture_revision) || value.capture_revision < 1 || value.capture_revision > value.revision)) return false;
  const fields = ['version', 'revision', 'status', ...(capture ? ['capture_revision'] : []), ...(status === 'countdown' ? ['seconds'] : status === 'preview' ? ['image'] : [])];
  if (Object.keys(value).length !== fields.length || !fields.every(key => Object.prototype.hasOwnProperty.call(value, key))) return false;
  if (status === 'countdown') return Number.isInteger(value.seconds) && value.seconds >= 1 && value.seconds <= 3;
  if (status === 'preview')
    return (
      typeof value.image === 'string' &&
      value.image.length > 0 &&
      value.image.length <= 666668 &&
      value.image.length % 4 === 0 &&
      /^[A-Za-z0-9+/]+={0,2}$/.test(value.image) &&
      value.image.startsWith('/9j/')
    );
  return true;
}

function christmasPhotoArguments(config) {
  const directory = config.christmasPhotoDirectory;
  return typeof directory === 'string' && directory.length > 0 ? ['--christmasPhotoDirectory=' + directory] : [];
}

module.exports = NodeHelper.create({
  pyshell: null,
  _workerState: null,

  worker_state: function () {
    if (!this._workerState) {
      this._workerState = { stopped: false, active: true, timer: null, attempts: 0, available: null, draining: false, startAfterClose: false, exitRegistered: false };
    }
    return this._workerState;
  },

  worker_availability: function (available) {
    const state = this.worker_state();
    if (state.available === available) return;
    state.available = available;
    this.sendSocketNotification('worker_state', { available });
  },

  cancel_worker_restart: function () {
    const state = this.worker_state();
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = null;
  },

  terminate_worker: function (shell) {
    const child = shell.childProcess;
    if (!child || child.killed || child.exitCode != null || child.signalCode != null) return;
    try {
      child.kill('SIGTERM');
    } catch (_error) {
      // Keep this process as the owner until a confirmed close; never overlap.
      console.error('[' + this.name + '] Worker termination failed; automatic replacement is waiting for process closure.');
    }
  },

  schedule_worker_restart: function () {
    const state = this.worker_state();
    if (state.stopped || !state.active || state.timer !== null || this.pyshell) return;
    const configuredLimit = this.config.workerRestartLimit ?? 3;
    const limit = Number.isInteger(configuredLimit) && configuredLimit >= 0 && configuredLimit <= 10 ? configuredLimit : 3;
    const configuredDelay = this.config.workerRestartDelay ?? 1000;
    const delay = Number.isInteger(configuredDelay) && configuredDelay >= 100 && configuredDelay <= 30000 ? configuredDelay : 1000;
    if (state.attempts >= limit) {
      if (limit > 0) console.error('[' + this.name + '] Worker recovery limit reached; check the camera and configuration before reloading.');
      return;
    }
    state.timer = setTimeout(
      () => {
        state.timer = null;
        if (state.stopped || !state.active || this.pyshell) return;
        state.attempts++;
        this.python_start();
      },
      Math.min(delay * Math.pow(2, state.attempts), 30000),
    );
  },

  python_start: function () {
    const self = this;
    const state = this.worker_state();
    if (!this.config || state.stopped || !state.active || this.pyshell) return;
    if (!state.exitRegistered) {
      onExit(function (_code, _signal) {
        self.destroy();
      });
      state.exitRegistered = true;
    }
    const extendedDataset = this.config.extendDataset ? 'True' : 'False';
    const options = {
      mode: 'json',
      formatter: 'text', // Commands are plain lines; worker stdout remains JSON.
      pythonOptions: ['-u'], // Immediately flush buffer for std out/in monitoring/writing to work
      stderrParser: line => JSON.stringify(line),
      args: [
        '--cascade=' + this.config.cascade,
        '--encodings=' + this.config.encodings,
        '--rotateCamera=' + this.config.rotateCamera,
        '--method=' + this.config.method,
        '--detectionMethod=' + this.config.detectionMethod,
        '--interval=' + this.config.checkInterval,
        '--output=' + this.config.output,
        '--outputmm=' + this.config.outputmm,
        '--extendDataset=' + extendedDataset,
        '--dataset=' + this.config.dataset,
        '--tolerance=' + this.config.tolerance,
        '--brightness=' + this.config.brightness,
        '--contrast=' + this.config.contrast,
        '--resolution=' + this.config.resolution,
        '--processWidth=' + this.config.processWidth,
        '--run-only-on-notification=' + (this.config.external_trigger_notification !== '' ? '1' : '0'),
        '--cameraLores=' + (this.config.cameraLores ? 'True' : 'False'),
        '--cameraLoresFormat=' + (this.config.cameraLoresFormat || 'YUV420'),
        '--opencvThreads=' + (this.config.opencvThreads ?? 0),
        '--cameraAutofocus=' + (this.config.cameraAutofocus || 'default'),
        '--cameraExposureMode=' + (this.config.cameraExposureMode || 'normal'),
        '--detectionUpsample=' + (this.config.detectionUpsample ?? 1),
        '--yunetModel=' + (this.config.yunetModel || 'modules/MMM-Face-Reco-DNN/model/face_detection_yunet_2023mar.onnx'),
        '--yunetScoreThreshold=' + (this.config.yunetScoreThreshold ?? 0.8),
        '--yunetNmsThreshold=' + (this.config.yunetNmsThreshold ?? 0.3),
        '--yunetTopK=' + (this.config.yunetTopK ?? 5000),
        '--identityMargin=' + (this.config.identityMargin ?? 0),
        '--previewQuality=' + (this.config.previewQuality ?? 95),
        ...christmasPhotoArguments(this.config),
      ],
    };

    if (this.config.cameraFrameRate != null) {
      options.args.push('--cameraFrameRate=' + this.config.cameraFrameRate);
    }

    if (this.config.cameraLensPosition != null) {
      options.args.push('--cameraLensPosition=' + this.config.cameraLensPosition);
    }

    if (this.config.pythonPath != null && this.config.pythonPath !== '') {
      options.pythonPath = this.config.pythonPath;
    }

    // A new process is started only after the previous process has closed.
    let shell;
    try {
      shell = new PythonShell('modules/' + this.name + '/tools/recognition.py', options);
    } catch (_error) {
      this.worker_availability(false);
      console.error('[' + this.name + '] Recognition worker could not start.');
      this.schedule_worker_restart();
      return;
    }
    self.pyshell = shell;
    state.draining = false;
    const startedAt = Date.now();
    let failed = false;
    const onFailure = function () {
      if (self.pyshell !== shell || failed || state.stopped) return;
      failed = true;
      state.draining = true;
      self.worker_availability(false);
      // Error messages can contain private gallery paths or labels.
      console.error('[' + self.name + '] Recognition worker failed; waiting for it to close before recovery.');
      self.terminate_worker(shell);
    };
    const onClose = function () {
      if (self.pyshell !== shell) return;
      self.pyshell = null;
      state.draining = false;
      self.worker_availability(false);
      if (state.stopped) return;
      // A long-running healthy session starts a fresh bounded recovery budget.
      if (Date.now() - startedAt >= 60000) state.attempts = 0;
      if (state.startAfterClose) {
        state.startAfterClose = false;
        self.python_start();
      } else {
        self.schedule_worker_restart();
      }
    };
    shell.on('error', onFailure);
    shell.on('pythonError', onFailure);
    shell.on('close', onClose);
    // Spawn failures do not always emit PythonShell's close event.
    if (shell.childProcess) shell.childProcess.once('close', onClose);

    // check if a message of the python script is comming in
    shell.on('message', function (message) {
      if (self.pyshell !== shell || failed || state.stopped || state.draining || !state.active || !message || typeof message !== 'object') return;
      if (message.ready === true) self.worker_availability(true);
      if (validPresence(message.presence)) {
        const value = message.presence;
        self.sendSocketNotification('presence', {
          version: 1,
          epoch: value.epoch,
          sequence: value.sequence,
          observed_at: value.observed_at,
          available: value.available,
          labels: value.labels.slice(),
        });
      }
      if (validChristmasPhoto(message.christmas_photo)) {
        self.sendSocketNotification('christmas_photo', { ...message.christmas_photo });
      }
      // A status message has received and will log
      if (Object.prototype.hasOwnProperty.call(message, 'status')) {
        console.log('[' + self.name + '] ' + message.status);
      }

      // Somebody new are in front of the camera, send it back to the Magic Mirror Module
      if (message.camera_image && typeof message.camera_image.image === 'string') {
        self.worker_availability(true);
        self.sendSocketNotification('camera_image', {
          image: message.camera_image.image,
        });
      }

      // Check if we get an image to show in the mirror
      if (message.login && Array.isArray(message.login.names)) {
        const names = message.login.names.filter(name => typeof name === 'string' && name.length > 0);
        self.worker_availability(true);
        console.log('[' + self.name + '] Face sessions opened: ' + names.length);
        self.sendSocketNotification('user', {
          action: 'login',
          users: names,
        });
      }

      // Somebody left the camera, send it back to the Magic Mirror Module
      if (message.logout && Array.isArray(message.logout.names)) {
        const names = message.logout.names.filter(name => typeof name === 'string' && name.length > 0);
        console.log('[' + self.name + '] Face sessions closed: ' + names.length);
        self.sendSocketNotification('user', {
          action: 'logout',
          users: names,
        });
      }
    });
  },

  send_python_cmd: function (cmd) {
    if (this.pyshell && !this.worker_state().stopped && !this.worker_state().draining) this.pyshell.send(cmd);
  },

  python_stop: function () {
    this.destroy();
  },

  destroy: function () {
    const state = this.worker_state();
    state.stopped = true;
    state.startAfterClose = false;
    this.cancel_worker_restart();
    this.worker_availability(false);
    const shell = this.pyshell;
    if (!shell) return;
    state.draining = true;
    this.terminate_worker(shell);
  },

  socketNotificationReceived: function (notification, payload) {
    // Configuration are received
    if (notification === 'CONFIG') {
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return;
      const state = this.worker_state();
      const wasStopped = state.stopped;
      this.config = payload;
      // Set static output to 0, because we do not need any output for MMM
      this.config.output = 0;
      state.stopped = false;
      state.attempts = 0;
      this.cancel_worker_restart();
      if (this.pyshell && (wasStopped || state.draining)) {
        state.startAfterClose = true;
      } else if (!this.pyshell) {
        this.python_start();
      }
    }

    // Notification for triggering face recognition received. Only send to python subprocess
    // if it has been started
    if (this.config && notification === this.config.external_trigger_notification) {
      const state = this.worker_state();
      state.active = payload === true;
      if (state.active) {
        this.send_python_cmd('start');
        if (!state.stopped && !this.pyshell) this.schedule_worker_restart();
      } else {
        this.cancel_worker_restart();
        this.worker_availability(false);
        this.send_python_cmd('stop');
      }
    }
  },

  stop: function () {
    this.python_stop();
  },
});
