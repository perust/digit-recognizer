/**
 * Holds the browser port to the behaviour of the Python original.
 *
 *     node --test tests/test_web.mjs
 *
 * Three things are checked, in increasing scope: that the network arithmetic
 * reproduces recorded outputs exactly enough, that preprocessing still obeys
 * MNIST's framing rules, and that the two together read the same ten strokes
 * the desktop app is tested with.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DigitModel, IMAGE_SIZE, DIGIT_BOX, INK_THRESHOLD, preprocess, segmentStrokes,
} from '../web/digit-model.js';

const PROJECT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const readJson = (relative) => JSON.parse(fs.readFileSync(path.join(PROJECT, relative), 'utf8'));
const readBuffer = (relative) => {
  const raw = fs.readFileSync(path.join(PROJECT, relative));
  // Detach from Node's pooled Buffer so typed-array views start at offset 0.
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
};

const model = new DigitModel(readJson('web/model.json'), readBuffer('web/weights.bin'));
const strokeFixture = readJson('tests/fixtures/strokes.json');

// ------------------------------------------------------------------ drawing

/**
 * Rasterise pen strokes the way a canvas would: round caps and joins, and
 * antialiased edges, approximated by sampling twice per pixel per axis.
 */
function draw(strokes, width, height, penWidth) {
  const scale = 2;
  const bigW = width * scale;
  const bigH = height * scale;
  const radius = (penWidth / 2) * scale;
  const mask = new Float32Array(bigW * bigH);

  for (const stroke of strokes) {
    for (let i = 0; i < stroke.length; i++) {
      const [ax, ay] = stroke[i].map((v) => v * scale);
      const [bx, by] = (stroke[i + 1] ?? stroke[i]).map((v) => v * scale);
      const minX = Math.max(0, Math.floor(Math.min(ax, bx) - radius));
      const maxX = Math.min(bigW - 1, Math.ceil(Math.max(ax, bx) + radius));
      const minY = Math.max(0, Math.floor(Math.min(ay, by) - radius));
      const maxY = Math.min(bigH - 1, Math.ceil(Math.max(ay, by) + radius));
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          if (distanceToSegment(x, y, ax, ay, bx, by) <= radius) mask[y * bigW + x] = 1;
        }
      }
    }
  }

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) total += mask[(y * scale + dy) * bigW + x * scale + dx];
      }
      out[y * width + x] = total / (scale * scale);
    }
  }
  return out;
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared));
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

const CANVAS = strokeFixture.canvasSize;

const strokesOf = (digit, transform = (point) => point) =>
  strokeFixture.strokes[String(digit)].map((stroke) => stroke.map(transform));

const drawDigit = (digit, transform = (point) => point) =>
  draw(strokesOf(digit, transform), CANVAS, CANVAS, strokeFixture.penWidth);

// The strip the browser writes on: wide enough for several digits in a row.
const STRIP_WIDTH = 640;
const STRIP_HEIGHT = 200;
const STRIP_SCALE = STRIP_HEIGHT / CANVAS;
const STRIP_PEN = strokeFixture.penWidth * STRIP_SCALE;
const DIGIT_PITCH = 120; // how far apart consecutive digits are placed

/** Lay digits along the strip the way a hand would, left to right. */
const placeAlongStrip = (digits) =>
  digits.flatMap((digit, index) =>
    strokesOf(digit, ([x, y]) => [
      x * STRIP_SCALE + index * DIGIT_PITCH - 20,
      y * STRIP_SCALE,
    ]).map((points) => ({ points })),
  );

function inkBox(image) {
  let top = IMAGE_SIZE, left = IMAGE_SIZE, bottom = -1, right = -1;
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      if (image[y * IMAGE_SIZE + x] <= INK_THRESHOLD) continue;
      top = Math.min(top, y); bottom = Math.max(bottom, y);
      left = Math.min(left, x); right = Math.max(right, x);
    }
  }
  return { height: bottom - top + 1, width: right - left + 1 };
}

function centerOfMass(image) {
  let total = 0, massY = 0, massX = 0;
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const value = image[y * IMAGE_SIZE + x];
      total += value; massY += value * y; massX += value * x;
    }
  }
  return { row: massY / total, column: massX / total };
}

// ------------------------------------------------------------------- tests

test('network reproduces the probabilities recorded from Python', () => {
  const fixture = readJson('tests/fixtures/model_parity.json');
  const pixels = Buffer.from(fixture.inputs, 'base64');
  let worst = 0;

  fixture.probabilities.forEach((expected, index) => {
    const image = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < image.length; i++) image[i] = pixels[index * image.length + i] / 255;

    const actual = model.predict(image);
    assert.equal(actual.indexOf(Math.max(...actual)), expected.indexOf(Math.max(...expected)),
      `sample ${index}: predicted digit differs from Python`);
    expected.forEach((value, digit) => {
      worst = Math.max(worst, Math.abs(actual[digit] - value));
    });
  });

  assert.ok(worst < 1e-3, `largest probability difference ${worst} should stay under 1e-3`);
});

test('preprocessing returns a 28x28 frame in range', () => {
  const image = preprocess(drawDigit(7), CANVAS, CANVAS);
  assert.equal(image.length, IMAGE_SIZE * IMAGE_SIZE);
  assert.ok(Math.min(...image) >= 0 && Math.max(...image) <= 1);
});

test('a blank canvas is reported as blank, not guessed at', () => {
  assert.equal(preprocess(new Float32Array(CANVAS * CANVAS), CANVAS, CANVAS), null);
  assert.equal(preprocess(new Float32Array(CANVAS * CANVAS).fill(1), CANVAS, CANVAS), null);
});

test('the digit is scaled into the 20x20 box', () => {
  for (const factor of [0.5, 1, 1.4]) {
    const image = preprocess(
      drawDigit(7, ([x, y]) => [(x - 150) * factor + 150, (y - 150) * factor + 150]),
      CANVAS, CANVAS,
    );
    const box = inkBox(image);
    assert.ok(Math.max(box.height, box.width) <= DIGIT_BOX, `scale ${factor}: ${box.height}x${box.width}`);
    assert.ok(Math.max(box.height, box.width) >= DIGIT_BOX - 2, `scale ${factor}: ${box.height}x${box.width}`);
  }
});

test('the centre of mass lands in the middle wherever the digit was drawn', () => {
  for (const [dx, dy] of [[0, 0], [-40, 30], [35, -25]]) {
    const image = preprocess(drawDigit(7, ([x, y]) => [x + dx, y + dy]), CANVAS, CANVAS);
    const { row, column } = centerOfMass(image);
    const middle = (IMAGE_SIZE - 1) / 2;
    assert.ok(Math.abs(row - middle) <= 1, `row ${row}`);
    assert.ok(Math.abs(column - middle) <= 1, `column ${column}`);
  }
});

test('dark ink on light paper is inverted to match the training data', () => {
  const drawn = drawDigit(7);
  const scanned = new Float32Array(drawn.length);
  for (let i = 0; i < drawn.length; i++) scanned[i] = 1 - drawn[i];

  const fromDrawing = preprocess(drawn, CANVAS, CANVAS);
  const fromScan = preprocess(scanned, CANVAS, CANVAS);
  for (let i = 0; i < fromDrawing.length; i++) {
    assert.ok(Math.abs(fromDrawing[i] - fromScan[i]) < 1e-6, `pixel ${i}`);
  }
});

test('faint pencil is stretched back to full contrast', () => {
  const drawn = drawDigit(7);
  const faint = new Float32Array(drawn.length);
  for (let i = 0; i < drawn.length; i++) faint[i] = drawn[i] * 0.35;

  const strong = preprocess(drawn, CANVAS, CANVAS);
  const weak = preprocess(faint, CANVAS, CANVAS);
  for (let i = 0; i < strong.length; i++) {
    assert.ok(Math.abs(strong[i] - weak[i]) < 0.05, `pixel ${i}`);
  }
});

test('reads all ten strokes the desktop app is tested with', () => {
  const misread = [];
  for (let digit = 0; digit <= 9; digit++) {
    const probabilities = model.predict(preprocess(drawDigit(digit), CANVAS, CANVAS));
    const predicted = probabilities.indexOf(Math.max(...probabilities));
    if (predicted !== digit) misread.push(`${digit} read as ${predicted}`);
  }
  assert.deepEqual(misread, [], 'every stroke should be read back correctly');
});

// -------------------------------------------------- writing several in a row

test('a digit written in two strokes stays one digit', () => {
  // A 4 is a diagonal and a bar that crosses it: separate strokes, one digit.
  for (const digit of [3, 4, 8, 9]) {
    const written = strokesOf(digit).map((points) => ({ points }));
    assert.ok(written.length > 1, `digit ${digit} should be a multi-stroke test case`);
    assert.equal(segmentStrokes(written).length, 1, `digit ${digit} was split apart`);
  }
});

test('digits written side by side are told apart, in writing order', () => {
  const groups = segmentStrokes(placeAlongStrip([2, 0, 2, 6]));
  assert.equal(groups.length, 4);
  for (let i = 1; i < groups.length; i++) {
    assert.ok(groups[i - 1].maxX < groups[i].minX, 'groups should come back left to right');
  }
});

test('a gap narrower than the threshold keeps neighbours together', () => {
  // Two bars a hair apart are one group; the same bars spaced out are two.
  const close = [{ points: [[100, 20], [100, 180]] }, { points: [[108, 20], [108, 180]] }];
  const apart = [{ points: [[100, 20], [100, 180]] }, { points: [[220, 20], [220, 180]] }];
  assert.equal(segmentStrokes(close).length, 1);
  assert.equal(segmentStrokes(apart).length, 2);
});

test('a number written across the strip reads back in order', () => {
  for (const expected of [[2, 0, 2, 6], [4, 1, 5], [7, 3, 9, 0, 8]]) {
    const groups = segmentStrokes(placeAlongStrip(expected));
    assert.equal(groups.length, expected.length, `expected ${expected.length} digits`);

    // Each group is rasterised on its own, exactly as the page does it.
    const read = groups.map((group) => {
      const ink = draw(group.strokes.map((s) => s.points), STRIP_WIDTH, STRIP_HEIGHT, STRIP_PEN);
      const probabilities = model.predict(preprocess(ink, STRIP_WIDTH, STRIP_HEIGHT));
      return probabilities.indexOf(Math.max(...probabilities));
    });
    assert.deepEqual(read, expected, `wrote ${expected.join('')}, read ${read.join('')}`);
  }
});
