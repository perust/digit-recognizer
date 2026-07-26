/**
 * The drawing board: pointer input in, a digit out.
 *
 * All the arithmetic lives in digit-model.js; this file only deals with the
 * canvas, the pointer and the readout.
 */

import { IMAGE_SIZE, fetchModel, preprocess } from './digit-model.js';

const PEN_WIDTH = 22; // matches the desktop app, so a stroke survives the shrink to 28px
const PREDICT_DELAY_MS = 150; // idle time before the prediction is refreshed

const pad = document.getElementById('pad');
const preview = document.getElementById('preview');
const digitLabel = document.getElementById('digit');
const confidenceLabel = document.getElementById('confidence');
const barList = document.getElementById('bars');
const verdict = document.getElementById('verdict');
const clearButton = document.getElementById('clear');

const context = pad.getContext('2d', { willReadFrequently: true });
const previewContext = preview.getContext('2d');

const rows = buildBars();
let model = null;
let pendingPrediction = 0;
let drawing = false;

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
  if (event.key === 'c' && !event.metaKey && !event.ctrlKey) clear();
});

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
