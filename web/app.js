/**
 * The drawing board: pointer input in, a digit out.
 *
 * All the arithmetic lives in digit-model.js; this file only deals with the
 * canvas, the pointer and the readout.
 */

import { IMAGE_SIZE, fetchModel, preprocess } from './digit-model.js';

const PEN_WIDTH = 22; // matches the desktop app, so a stroke survives the shrink to 28px
const PREDICT_DELAY_MS = 150; // idle time before the prediction is refreshed

// How long the pen has to rest before the digit is added on its own. Short
// enough that writing a number does not feel like waiting; the cost is that a
// two-stroke digit -- 4, 5, a crossed 7 -- is filed early if the pointer takes
// longer than this to reach the second stroke, turning a "4" into "1" then "1".
// Starting a stroke calls a pending add off and Backspace takes one back, which
// is what keeps that recoverable rather than merely annoying.
const AUTO_COMMIT_MS = 600;
const AUTO_COMMIT_KEY = 'digit-recognizer:auto-commit';

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
const autoLabel = document.getElementById('auto-label');
const countdown = document.getElementById('countdown');

autoLabel.textContent = `Add on its own after a ${(AUTO_COMMIT_MS / 1000).toFixed(1)} second pause`;

const context = pad.getContext('2d', { willReadFrequently: true });
const previewContext = preview.getContext('2d');

const rows = buildBars();
let model = null;
let pendingPrediction = 0;
let drawing = false;
let reading = null; // the digit the pad currently shows, or null when it is blank
let autoTimer = 0;

autoToggle.checked = remembered(AUTO_COMMIT_KEY) !== 'off';

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

context.strokeStyle = '#fff';
context.lineWidth = PEN_WIDTH;
context.lineCap = 'round';
context.lineJoin = 'round';

function clear() {
  cancelAutoCommit();
  context.fillStyle = '#000';
  context.fillRect(0, 0, pad.width, pad.height);
  // Back to the pen colour straight away: the same fillStyle draws the dot a
  // tap leaves behind, and forgetting this made every dot after a Clear black.
  context.fillStyle = '#fff';
  showBlank();
}

clear();

fetchModel(new URL('./', import.meta.url).href)
  .then((loaded) => {
    model = loaded;
    confidenceLabel.textContent = 'nothing drawn yet';
  })
  .catch((error) => {
    confidenceLabel.textContent = 'could not load the model';
    console.error(error);
  });

// ------------------------------------------------------------------- drawing

/** Pointer coordinates are in CSS pixels; the canvas may be displayed smaller. */
function canvasPoint(event) {
  const box = pad.getBoundingClientRect();
  return {
    x: ((event.clientX - box.left) / box.width) * pad.width,
    y: ((event.clientY - box.top) / box.height) * pad.height,
  };
}

pad.addEventListener('pointerdown', (event) => {
  drawing = true;
  cancelAutoCommit(); // a new stroke means the digit is not finished
  pad.setPointerCapture(event.pointerId);
  const { x, y } = canvasPoint(event);
  // A tap with no drag should still leave a mark.
  context.beginPath();
  context.arc(x, y, PEN_WIDTH / 2, 0, Math.PI * 2);
  context.fill();
  context.beginPath();
  context.moveTo(x, y);
  schedulePrediction();
});

pad.addEventListener('pointermove', (event) => {
  if (!drawing) return;
  const { x, y } = canvasPoint(event);
  context.lineTo(x, y);
  context.stroke();
  context.beginPath();
  context.moveTo(x, y);
  schedulePrediction();
});

for (const type of ['pointerup', 'pointercancel', 'pointerleave']) {
  pad.addEventListener(type, () => {
    if (!drawing) return;
    drawing = false;
    schedulePrediction();
  });
}

clearButton.addEventListener('click', clear);

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey) return; // leave Cmd+C and the rest alone
  if (event.key === 'Enter') {
    event.preventDefault();
    commitDigit();
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

// -------------------------------------------------------- automatic adding

/** Begin the wait after which a resting digit adds itself. */
function startAutoCommit() {
  cancelAutoCommit();
  countdown.style.transition = 'none';
  countdown.style.width = '0%';
  void countdown.offsetWidth; // flush, or the reset is folded into the animation
  countdown.style.transition = `width ${AUTO_COMMIT_MS}ms linear`;
  countdown.style.width = '100%';
  autoTimer = setTimeout(commitDigit, AUTO_COMMIT_MS);
}

function cancelAutoCommit() {
  clearTimeout(autoTimer);
  autoTimer = 0;
  countdown.style.transition = 'none';
  countdown.style.width = '0%';
}

autoToggle.addEventListener('change', () => {
  remember(AUTO_COMMIT_KEY, autoToggle.checked ? 'on' : 'off');
  if (autoToggle.checked && reading !== null && !drawing) startAutoCommit();
  else if (!autoToggle.checked) cancelAutoCommit();
});

// ------------------------------------------------------------ collected text

/** Append what the pad reads and make room for the next digit. */
function commitDigit() {
  if (reading === null) return;
  output.value += String(reading);
  clear();
}

/**
 * Drop the digit added last. The safety net for the automatic add: when a
 * two-stroke digit gets filed halfway through, one press takes it back.
 */
function undoDigit() {
  output.value = output.value.slice(0, -1);
}

addButton.addEventListener('click', commitDigit);
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

// ---------------------------------------------------------------- prediction

/** Debounce: only classify once the pen has been still for a moment. */
function schedulePrediction() {
  clearTimeout(pendingPrediction);
  pendingPrediction = setTimeout(predict, PREDICT_DELAY_MS);
}

function predict() {
  if (!model) return;
  const image = preprocess(readCanvas(), pad.width, pad.height);
  if (image === null) {
    showBlank();
    return;
  }
  showResult(model.predict(image));
  showPreview(image);
  // Only once the pen is up: holding still mid-stroke is not a finished digit.
  if (autoToggle.checked && !drawing) startAutoCommit();
}

/** The pad only ever holds white on black, so any channel is the intensity. */
function readCanvas() {
  const { data } = context.getImageData(0, 0, pad.width, pad.height);
  const gray = new Float32Array(pad.width * pad.height);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4] / 255;
  return gray;
}

function showResult(probabilities) {
  let best = 0;
  for (let digit = 1; digit < probabilities.length; digit++) {
    if (probabilities[digit] > probabilities[best]) best = digit;
  }

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
  confidenceLabel.textContent = model ? 'nothing drawn yet' : 'loading the model…';
  rows.forEach(({ row, fill, value }) => {
    fill.style.width = '0%';
    value.textContent = '0.0%';
    row.classList.remove('top');
  });
  showPreview(new Float32Array(IMAGE_SIZE * IMAGE_SIZE));
}
