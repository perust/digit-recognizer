/**
 * Browser-side twin of digit_recognizer/preprocess.py and the trained CNN.
 *
 * No framework: the exported plan only contains convolution, max-pool and
 * dense steps, and at this size plain loops finish in a couple of milliseconds.
 * The point of the file is that the arithmetic here has to agree with the
 * Python original -- tests/test_web.mjs replays the same strokes through both.
 */

export const IMAGE_SIZE = 28; // side of the frame the network expects
export const DIGIT_BOX = 20; // side of the box the digit is scaled to fit
export const INK_THRESHOLD = 0.12; // above this, on a 0..1 scale, counts as ink

// ---------------------------------------------------------------- preprocess

/**
 * Normalise a drawing into MNIST's format: a 20x20 digit centred by mass on a
 * 28x28 frame. Returns null when the picture holds no strokes at all.
 *
 * @param {Float32Array} gray pixels in [0, 1], row major
 */
export function preprocess(gray, width, height) {
  const levelled = normaliseLevels(ensureWhiteOnBlack(gray, width, height), width, height);
  const box = inkBounds(levelled, width, height);
  if (box === null) return null;

  const cropped = crop(levelled, width, box);
  const scale = DIGIT_BOX / Math.max(box.height, box.width);
  // Clamp to one pixel so a thin, tall "1" cannot collapse to zero width.
  const fittedW = Math.max(1, Math.round(box.width * scale));
  const fittedH = Math.max(1, Math.round(box.height * scale));
  const fitted = resizeArea(cropped, box.width, box.height, fittedW, fittedH);

  return centerByMass(fitted, fittedW, fittedH);
}

/** Invert when the picture holds dark ink on a light page, as a photo does. */
function ensureWhiteOnBlack(gray, width, height) {
  if (borderMedian(gray, width, height) <= 0.5) return gray;
  const flipped = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) flipped[i] = 1 - gray[i];
  return flipped;
}

/** Push the background to pure black and the strongest stroke to pure white. */
function normaliseLevels(gray, width, height) {
  const background = borderMedian(gray, width, height);
  const cleaned = new Float32Array(gray.length);
  let peak = 0;
  for (let i = 0; i < gray.length; i++) {
    const value = Math.min(1, Math.max(0, gray[i] - background));
    cleaned[i] = value;
    if (value > peak) peak = value;
  }
  if (peak > 0) for (let i = 0; i < cleaned.length; i++) cleaned[i] /= peak;
  return cleaned;
}

/** The border of a picture is almost always background, so measure it there. */
function borderMedian(gray, width, height) {
  const edge = [];
  for (let x = 0; x < width; x++) {
    edge.push(gray[x], gray[(height - 1) * width + x]);
  }
  for (let y = 0; y < height; y++) {
    edge.push(gray[y * width], gray[y * width + width - 1]);
  }
  edge.sort((a, b) => a - b);
  const middle = edge.length >> 1;
  return edge.length % 2 ? edge[middle] : (edge[middle - 1] + edge[middle]) / 2;
}

/** Tight box around the strokes, or null when nothing was drawn. */
function inkBounds(gray, width, height) {
  let top = height, left = width, bottom = -1, right = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x] <= INK_THRESHOLD) continue;
      if (y < top) top = y;
      if (y > bottom) bottom = y;
      if (x < left) left = x;
      if (x > right) right = x;
    }
  }
  if (bottom < 0) return null;
  return { top, left, width: right - left + 1, height: bottom - top + 1 };
}

function crop(gray, width, box) {
  const out = new Float32Array(box.width * box.height);
  for (let y = 0; y < box.height; y++) {
    const from = (box.top + y) * width + box.left;
    out.set(gray.subarray(from, from + box.width), y * box.width);
  }
  return out;
}

/**
 * Resample by averaging over the source area each destination pixel covers.
 * The canvas is always far larger than the 20 pixel box, so this only ever
 * runs as a downscale, where area averaging matches the anti-aliased shrink
 * the Python side gets from Pillow without any of its ringing.
 */
function resizeArea(src, srcW, srcH, dstW, dstH) {
  const out = new Float32Array(dstW * dstH);
  const stepX = srcW / dstW;
  const stepY = srcH / dstH;
  for (let dy = 0; dy < dstH; dy++) {
    const y0 = dy * stepY;
    const y1 = (dy + 1) * stepY;
    for (let dx = 0; dx < dstW; dx++) {
      const x0 = dx * stepX;
      const x1 = (dx + 1) * stepX;
      // Clamped because dstH * (srcH / dstH) can land a hair above srcH, and
      // one row past the end reads undefined, which turns the image into NaN.
      const lastY = Math.min(srcH, Math.ceil(y1));
      const lastX = Math.min(srcW, Math.ceil(x1));
      let total = 0;
      let weight = 0;
      for (let sy = Math.floor(y0); sy < lastY; sy++) {
        const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
        for (let sx = Math.floor(x0); sx < lastX; sx++) {
          const w = wy * (Math.min(x1, sx + 1) - Math.max(x0, sx));
          total += src[sy * srcW + sx] * w;
          weight += w;
        }
      }
      out[dy * dstW + dx] = weight > 0 ? total / weight : 0;
    }
  }
  return out;
}

/**
 * Lay the digit on the 28x28 frame with its centre of mass in the middle.
 * Centring by mass rather than by bounding box is what the original dataset
 * did, and the two differ by several pixels on a digit like 7.
 */
function centerByMass(digit, width, height) {
  const size = IMAGE_SIZE;
  const canvas = new Float32Array(size * size);
  const top = (size - height) >> 1;
  const left = (size - width) >> 1;
  for (let y = 0; y < height; y++) {
    canvas.set(digit.subarray(y * width, (y + 1) * width), (top + y) * size + left);
  }

  let total = 0;
  let massY = 0;
  let massX = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = canvas[y * size + x];
      total += value;
      massY += value * y;
      massX += value * x;
    }
  }
  if (total === 0) return canvas;

  const middle = (size - 1) / 2;
  return translate(canvas, size, Math.round(middle - massX / total), Math.round(middle - massY / total));
}

/** Shift by whole pixels, leaving the vacated border as background. */
function translate(image, size, dx, dy) {
  const out = new Float32Array(image.length);
  for (let y = 0; y < size; y++) {
    const sourceY = y - dy;
    if (sourceY < 0 || sourceY >= size) continue;
    for (let x = 0; x < size; x++) {
      const sourceX = x - dx;
      if (sourceX < 0 || sourceX >= size) continue;
      out[y * size + x] = image[sourceY * size + sourceX];
    }
  }
  return out;
}

// ----------------------------------------------------------------- inference

export class DigitModel {
  /**
   * @param {object} spec parsed model.json
   * @param {ArrayBuffer} buffer contents of weights.bin
   */
  constructor(spec, buffer) {
    this.spec = spec;
    // Dequantised once here rather than on every prediction.
    this.tensors = spec.tensors.map((tensor) => readTensor(tensor, buffer));
  }

  /**
   * @param {Float32Array} image 28*28 pixels in [0, 1]
   * @returns {Float32Array} probability of each digit
   */
  predict(image) {
    let values = image;
    let height = this.spec.input[0];
    let width = this.spec.input[1];
    let channels = this.spec.input[2];

    for (const step of this.spec.layers) {
      if (step.type === 'conv') {
        const [, , , outChannels] = step.shape;
        values = conv2d(values, height, width, channels, this.tensors[step.weights], outChannels, this.tensors[step.bias]);
        channels = outChannels;
      } else if (step.type === 'maxpool') {
        values = maxPool(values, height, width, channels, step.size);
        height = Math.floor(height / step.size);
        width = Math.floor(width / step.size);
      } else if (step.type === 'flatten') {
        // Already row major over (row, column, channel), which is the order
        // Keras's Flatten produces, so there is nothing to rearrange.
        height = width = 1;
      } else if (step.type === 'dense') {
        values = dense(values, this.tensors[step.weights], step.shape[0], step.shape[1], this.tensors[step.bias]);
        channels = step.shape[1];
      }

      if (step.activation === 'relu') relu(values);
      else if (step.activation === 'softmax') values = softmax(values);
    }
    return values;
  }
}

/** Read one tensor out of the blob, undoing the per-channel int8 scaling. */
function readTensor(tensor, buffer) {
  if (tensor.dtype === 'float32') {
    return new Float32Array(buffer, tensor.offset, tensor.count);
  }
  const raw = new Int8Array(buffer, tensor.offset, tensor.count);
  const out = new Float32Array(tensor.count);
  // One scale per output channel, and the output channel is the fastest
  // moving axis of every tensor we export, so it is just index modulo width.
  const scales = tensor.scales;
  const stride = scales.length;
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] * scales[i % stride];
  return out;
}

/** 3x3 convolution, stride 1, zero padded to keep the resolution ("same"). */
function conv2d(input, height, width, inChannels, kernel, outChannels, bias) {
  const out = new Float32Array(height * width * outChannels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const target = (y * width + x) * outChannels;
      for (let c = 0; c < outChannels; c++) out[target + c] = bias[c];

      for (let ky = 0; ky < 3; ky++) {
        const sourceY = y + ky - 1;
        if (sourceY < 0 || sourceY >= height) continue;
        for (let kx = 0; kx < 3; kx++) {
          const sourceX = x + kx - 1;
          if (sourceX < 0 || sourceX >= width) continue;
          const source = (sourceY * width + sourceX) * inChannels;
          const plane = (ky * 3 + kx) * inChannels * outChannels;
          for (let ic = 0; ic < inChannels; ic++) {
            const value = input[source + ic];
            if (value === 0) continue; // most of a digit is background
            const row = plane + ic * outChannels;
            for (let oc = 0; oc < outChannels; oc++) out[target + oc] += value * kernel[row + oc];
          }
        }
      }
    }
  }
  return out;
}

function maxPool(input, height, width, channels, size) {
  const outH = Math.floor(height / size);
  const outW = Math.floor(width / size);
  const out = new Float32Array(outH * outW * channels);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const target = (y * outW + x) * channels;
      for (let c = 0; c < channels; c++) out[target + c] = -Infinity;
      for (let dy = 0; dy < size; dy++) {
        for (let dx = 0; dx < size; dx++) {
          const source = ((y * size + dy) * width + x * size + dx) * channels;
          for (let c = 0; c < channels; c++) {
            if (input[source + c] > out[target + c]) out[target + c] = input[source + c];
          }
        }
      }
    }
  }
  return out;
}

function dense(input, kernel, inDim, outDim, bias) {
  const out = new Float32Array(outDim);
  out.set(bias);
  for (let i = 0; i < inDim; i++) {
    const value = input[i];
    if (value === 0) continue;
    const row = i * outDim;
    for (let o = 0; o < outDim; o++) out[o] += value * kernel[row + o];
  }
  return out;
}

function relu(values) {
  for (let i = 0; i < values.length; i++) if (values[i] < 0) values[i] = 0;
}

function softmax(values) {
  const out = new Float32Array(values.length);
  let peak = -Infinity;
  for (const value of values) if (value > peak) peak = value;
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    out[i] = Math.exp(values[i] - peak);
    total += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

/** Convenience loader for the browser; Node builds a DigitModel directly. */
export async function fetchModel(baseUrl = './') {
  const [spec, buffer] = await Promise.all([
    fetch(`${baseUrl}model.json`).then((response) => response.json()),
    fetch(`${baseUrl}weights.bin`).then((response) => response.arrayBuffer()),
  ]);
  return new DigitModel(spec, buffer);
}
