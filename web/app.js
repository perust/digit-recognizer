/**
 * The writing strip: pointer input in, a number out.
 *
 * Digits are written one after another across the strip. A digit is taken as
 * finished the moment the next one is started to its right, and otherwise once
 * the pen has rested; its ink then fades out, which is the only signal that it
 * has been read and is no longer part of what the network is looking at.
 *
 * All the arithmetic lives in digit-model.js. Strokes are kept as points rather
 * than painted straight onto the canvas, because they have to be redrawn every
 * frame at their own opacity and rasterised one digit at a time for recognition.
 */

import { IMAGE_SIZE, fetchModel, preprocess, segmentStrokes } from './digit-model.js';

const PEN_WIDTH = 16; // proportioned to the strip, so a stroke survives the shrink to 28px
const PREDICT_DELAY_MS = 150; // idle time before the reading is refreshed

// How long the pen has to rest before the last digit is taken. Only the last
// one waits: everything before it was already settled by the pen moving on.
// The reader can change it, because the right value depends on how fast they
// write; the bounds only keep it from being set to something unusable.
const DEFAULT_AUTO_COMMIT_MS = 800;
const MIN_AUTO_COMMIT_MS = 200;
const MAX_AUTO_COMMIT_MS = 5000;
const FADE_MS = 450; // how long collected ink takes to disappear
const AUTO_COMMIT_KEY = 'digit-recognizer:auto-commit';
const PAUSE_KEY = 'digit-recognizer:pause-seconds';

const pad = document.getElementById('pad');
const preview = document.getElementById('preview');
const digitLabel = document.getElementById('digit');
const confidenceLabel = document.getElementById('confidence');
const barList = document.getElementById('bars');
const verdict = document.getElementById('verdict');
const clearButton = document.getElementById('clear');
const output = document.getElementById('output');
const addButton = document.getElementById('add');
const undoButton = document.getElementById('undo');
const copyButton = document.getElementById('copy');
const wipeButton = document.getElementById('wipe');
const autoToggle = document.getElementById('auto');
const delayInput = document.getElementById('delay');
const countdown = document.getElementById('countdown');

const context = pad.getContext('2d', { willReadFrequently: true });
const previewContext = preview.getContext('2d');

// One digit at a time is rasterised here for the network, away from the ink
// the reader can see, so fading strokes never reach the model.
const stage = document.createElement('canvas');
stage.width = pad.width;
stage.height = pad.height;
const stageContext = stage.getContext('2d', { willReadFrequently: true });

const rows = buildBars();
let model = null;
let strokes = []; // { points: [[x, y], ...], fadeStart: number | null }
let stroke = null; // the one being drawn right now
let reading = null; // what the rightmost unread digit says, or null when there is none
let pendingPrediction = 0;
let autoTimer = 0;
let frame = 0;
let autoCommitMs = DEFAULT_AUTO_COMMIT_MS;

autoToggle.checked = remembered(AUTO_COMMIT_KEY) !== 'off';
applyPause(remembered(PAUSE_KEY) ?? String(DEFAULT_AUTO_COMMIT_MS / 1000));

/**
 * Take whatever was typed into the pause field, hold it to something usable,
 * and write the accepted value back so the field never disagrees with the timer.
 */
function applyPause(text) {
  const seconds = Number.parseFloat(text);
  autoCommitMs = Number.isFinite(seconds)
    ? Math.min(MAX_AUTO_COMMIT_MS, Math.max(MIN_AUTO_COMMIT_MS, Math.round(seconds * 1000)))
    : DEFAULT_AUTO_COMMIT_MS;
  delayInput.value = (autoCommitMs / 1000).toFixed(1);
  remember(PAUSE_KEY, delayInput.value);
}

// "change" rather than "input": rewriting the field on every keystroke would
// clamp "0" to the minimum before "0.8" had been finished.
delayInput.addEventListener('change', () => {
  applyPause(delayInput.value);
  if (autoTimer) startAutoCommit(); // let a running countdown show the new length
});

/** Storage throws rather than degrading in some privacy modes, so ask carefully. */
function remembered(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // A preference that cannot be stored is not worth failing over.
  }
}

// --------------------------------------------------------------------- setup

function buildBars() {
  return Array.from({ length: 10 }, (_, digit) => {
    const row = document.createElement('li');
    row.innerHTML = `<b>${digit}</b><span class="track"><span class="fill"></span></span><span class="value">0.0%</span>`;
    barList.append(row);
    return { row, fill: row.querySelector('.fill'), value: row.querySelector('.value') };
  });
}

/** Resizing a canvas resets everything about its context, so set the pen here. */
function preparePens() {
  for (const target of [context, stageContext]) {
    target.lineWidth = PEN_WIDTH;
    target.lineCap = 'round';
    target.lineJoin = 'round';
  }
}

/**
 * Match the backing store to however wide the page is.  A narrow phone then
 * gets a shorter strip rather than a squashed one, and because the two stay in
 * step there is exactly one canvas pixel per CSS pixel.
 */
function fitStrip() {
  const width = Math.max(280, Math.round(pad.getBoundingClientRect().width));
  if (width === pad.width) return;
  pad.width = width;
  stage.width = width;
  preparePens();
  render();
}

preparePens();
fitStrip();
window.addEventListener('resize', fitStrip);

function clear() {
  cancelAutoCommit();
  strokes = [];
  stroke = null;
  render();
  showBlank();
}

clear();

fetchModel(new URL('./', import.meta.url).href)
  .then((loaded) => {
    model = loaded;
    confidenceLabel.textContent = 'nothing written yet';
  })
  .catch((error) => {
    confidenceLabel.textContent = 'could not load the model';
    console.error(error);
  });

// ------------------------------------------------------------------ painting

function paintStroke(target, path, alpha) {
  target.globalAlpha = alpha;
  target.strokeStyle = '#fff';
  target.fillStyle = '#fff';
  const [head, ...tail] = path.points;
  if (tail.length === 0) {
    // A tap with no drag should still leave a mark.
    target.beginPath();
    target.arc(head[0], head[1], PEN_WIDTH / 2, 0, Math.PI * 2);
    target.fill();
  } else {
    target.beginPath();
    target.moveTo(head[0], head[1]);
    for (const [x, y] of tail) target.lineTo(x, y);
    target.stroke();
  }
  target.globalAlpha = 1;
}

function render() {
  frame = 0;
  context.fillStyle = '#000';
  context.fillRect(0, 0, pad.width, pad.height);

  const now = performance.now();
  strokes = strokes.filter((path) => path.fadeStart === null || now - path.fadeStart < FADE_MS);

  let fading = false;
  for (const path of strokes) {
    let alpha = 1;
    if (path.fadeStart !== null) {
      alpha = 1 - (now - path.fadeStart) / FADE_MS;
      fading = true;
    }
    paintStroke(context, path, alpha);
  }
  if (fading) scheduleFrame();
}

function scheduleFrame() {
  if (!frame) frame = requestAnimationFrame(render);
}

/** Everything still waiting to be read: what has faded out is already collected. */
function unread() {
  return strokes.filter((path) => path.fadeStart === null);
}

/** Rasterise one digit on its own, at full opacity, for the network. */
function imageOf(group) {
  stageContext.fillStyle = '#000';
  stageContext.fillRect(0, 0, stage.width, stage.height);
  for (const path of group) paintStroke(stageContext, path, 1);

  const { data } = stageContext.getImageData(0, 0, stage.width, stage.height);
  const gray = new Float32Array(stage.width * stage.height);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4] / 255;
  return preprocess(gray, stage.width, stage.height);
}

// ------------------------------------------------------------------- writing

/** Pointer coordinates are in CSS pixels; the strip may be displayed smaller. */
function stripPoint(event) {
  const box = pad.getBoundingClientRect();
  return [
    ((event.clientX - box.left) / box.width) * pad.width,
    ((event.clientY - box.top) / box.height) * pad.height,
  ];
}

pad.addEventListener('pointerdown', (event) => {
  cancelAutoCommit();
  pad.setPointerCapture(event.pointerId);
  stroke = { points: [stripPoint(event)], fadeStart: null };
  strokes.push(stroke);

  // Starting a digit settles whatever lies to its left. Re-segmenting with the
  // new stroke included is what decides that: a second stroke landing on the
  // digit already there joins its group and settles nothing, while one placed
  // further along starts a group of its own and closes the previous digit.
  const groups = segmentStrokes(unread());
  const here = groups.find((group) => group.strokes.includes(stroke));
  collect(groups.filter((group) => group !== here && group.maxX < here.minX));

  render();
  schedulePrediction();
});

pad.addEventListener('pointermove', (event) => {
  if (!stroke) return;
  stroke.points.push(stripPoint(event));
  render();
  schedulePrediction();
});

for (const type of ['pointerup', 'pointercancel', 'pointerleave']) {
  pad.addEventListener(type, () => {
    if (!stroke) return;
    stroke = null;
    schedulePrediction();
  });
}

clearButton.addEventListener('click', clear);

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey) return; // leave Cmd+C and the rest alone
  if (event.target === delayInput) return; // the pause field handles its own keys
  if (event.key === 'Enter') {
    event.preventDefault();
    collectEverything();
    return;
  }
  // Inside the field the keyboard is ordinary text editing; the shortcuts
  // below only apply everywhere else.
  if (event.target === output) return;
  if (event.key === 'Backspace') {
    event.preventDefault();
    undoDigit();
  } else if (event.key === 'c') {
    clear();
  }
});

// --------------------------------------------------------------- collecting

/** Read each group, append it, and start its ink fading. */
function collect(groups) {
  if (!model || groups.length === 0) return;
  const now = performance.now();
  let added = '';

  for (const group of groups) {
    const image = imageOf(group.strokes);
    if (image !== null) {
      const probabilities = model.predict(image);
      added += String(likeliest(probabilities));
    }
    for (const path of group.strokes) path.fadeStart = now;
  }

  output.value += added;
  scheduleFrame();
}

/** Everything left on the strip, in writing order. Used by Enter and the pause. */
function collectEverything() {
  cancelAutoCommit();
  const waiting = unread();
  if (waiting.length === 0) return;
  collect(segmentStrokes(waiting));
  showBlank();
}

/**
 * Drop the digit added last. The safety net for reading a digit too early:
 * one press takes it back.
 */
function undoDigit() {
  output.value = output.value.slice(0, -1);
}

addButton.addEventListener('click', collectEverything);
undoButton.addEventListener('click', undoDigit);
wipeButton.addEventListener('click', () => {
  output.value = '';
  output.focus();
});

copyButton.addEventListener('click', async () => {
  if (!output.value) return;
  try {
    await navigator.clipboard.writeText(output.value);
  } catch {
    // No async clipboard, or the page is not on a secure origin.
    output.select();
    document.execCommand('copy');
  }
  flash(copyButton, 'Copied');
});

/** Say something happened on the button itself, then put the label back. */
function flash(button, message) {
  const original = button.textContent;
  button.textContent = message;
  setTimeout(() => { button.textContent = original; }, 1200);
}

// ---------------------------------------------------------- the last digit

/** Begin the wait after which the digit still on the strip is taken. */
function startAutoCommit() {
  cancelAutoCommit();
  countdown.style.transition = 'none';
  countdown.style.width = '0%';
  void countdown.offsetWidth; // flush, or the reset is folded into the animation
  countdown.style.transition = `width ${autoCommitMs}ms linear`;
  countdown.style.width = '100%';
  autoTimer = setTimeout(collectEverything, autoCommitMs);
}

function cancelAutoCommit() {
  clearTimeout(autoTimer);
  autoTimer = 0;
  countdown.style.transition = 'none';
  countdown.style.width = '0%';
}

autoToggle.addEventListener('change', () => {
  remember(AUTO_COMMIT_KEY, autoToggle.checked ? 'on' : 'off');
  if (autoToggle.checked && reading !== null && !stroke) startAutoCommit();
  else if (!autoToggle.checked) cancelAutoCommit();
});

// ---------------------------------------------------------------- prediction

/** Debounce: only classify once the pen has been still for a moment. */
function schedulePrediction() {
  clearTimeout(pendingPrediction);
  pendingPrediction = setTimeout(predict, PREDICT_DELAY_MS);
}

function predict() {
  if (!model) return;
  const groups = segmentStrokes(unread());
  if (groups.length === 0) {
    showBlank();
    return;
  }

  // The rightmost group is the digit being written; the readout follows it.
  const image = imageOf(groups[groups.length - 1].strokes);
  if (image === null) {
    showBlank();
    return;
  }
  showResult(model.predict(image));
  showPreview(image);
  // Only once the pen is up: holding still mid-stroke is not a finished digit.
  if (autoToggle.checked && !stroke) startAutoCommit();
}

function likeliest(probabilities) {
  let best = 0;
  for (let digit = 1; digit < probabilities.length; digit++) {
    if (probabilities[digit] > probabilities[best]) best = digit;
  }
  return best;
}

function showResult(probabilities) {
  const best = likeliest(probabilities);
  reading = best;
  verdict.classList.remove('is-blank');
  digitLabel.textContent = String(best);
  confidenceLabel.textContent = `${(probabilities[best] * 100).toFixed(1)}% confident`;
  rows.forEach(({ row, fill, value }, digit) => {
    fill.style.width = `${probabilities[digit] * 100}%`;
    value.textContent = `${(probabilities[digit] * 100).toFixed(1)}%`;
    row.classList.toggle('top', digit === best);
  });
}

function showPreview(image) {
  const frame = previewContext.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let i = 0; i < image.length; i++) {
    const level = Math.round(image[i] * 255);
    frame.data.set([level, level, level, 255], i * 4);
  }
  previewContext.putImageData(frame, 0, 0);
}

function showBlank() {
  reading = null;
  verdict.classList.add('is-blank');
  digitLabel.textContent = '–';
  confidenceLabel.textContent = model ? 'nothing written yet' : 'loading the model…';
  rows.forEach(({ row, fill, value }) => {
    fill.style.width = '0%';
    value.textContent = '0.0%';
    row.classList.remove('top');
  });
  showPreview(new Float32Array(IMAGE_SIZE * IMAGE_SIZE));
}
