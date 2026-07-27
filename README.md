# Handwritten Digit Recognizer

Draw a digit and a convolutional network trained on MNIST reads it back. It
comes in three shapes: a desktop app, a web page that runs the network in the
browser, and a command line tool for images of digits written on paper.

```
digit_recognizer/
  preprocess.py   turns any picture of one digit into MNIST's exact format
  model.py        the CNN architecture
  train.py        trains on MNIST, saves models/digit_cnn.keras
  predict.py      command line classifier for image files
  app.py          tkinter drawing board with live predictions
web/
  index.html      the browser version: a strip you write a whole number across
  digit-model.js  the same preprocessing and network, ported to JavaScript
  app.js          canvas drawing and the readout
  model.json      layer plan and quantisation scales, written by the exporter
  weights.bin     the network as int8, 852 KB
tools/
  export_samples.py   writes MNIST test digits out as PNGs, for a sanity check
  export_web_model.py folds batch-norm away, quantises, writes web/weights.bin
  simulate_drawing.py drives the desktop GUI with synthetic mouse strokes
  make_launcher.py    builds "Digit Recognizer.app", the double-clickable launcher
tests/
  test_preprocess.py  the normalisation contract the model relies on
  test_web.mjs        holds the JavaScript port to the Python original
```

Trained for 12 epochs it reaches **99.50%** on the MNIST test set (**99.51%**
after the int8 quantisation the web build uses), classifies all 40 exported
sample files correctly, and reads back all ten synthetic strokes on both the
desktop app and the web page.

## Setup

```bash
pip3 install -r requirements.txt
```

## Train

```bash
python3 -m digit_recognizer.train            # 12 epochs, ~1 minute per epoch on a CPU
python3 -m digit_recognizer.train --epochs 20 --batch-size 256
```

MNIST is downloaded automatically (~11 MB) and cached in `~/.keras/datasets/`.
The best epoch by validation accuracy is written to `models/digit_cnn.keras`.

## Draw

Double-click **Digit Recognizer.app** in this folder, or from a shell:

```bash
python3 -m digit_recognizer.app
```

The prediction refreshes a moment after you stop moving the pen. The panel on
the right shows the probability of every digit and a magnified view of the
28x28 image the network actually receives — if that preview looks wrong, the
prediction will be too. Press `c` to clear.

Rest the pen and the digit adds itself to the field along the bottom, wiping
the pad for the next one, so a longer number is written a digit at a time. A
bar under the pad fills up during the wait — drawing again calls it off. Enter
adds the digit immediately, Backspace takes the last one back, the checkbox
turns the automatic add off entirely, and the wait itself is a field you can
type into.

That field is an ordinary text entry: correct a misread digit by hand, type into
it directly, select part of it, or hit Copy to put the whole thing on the
clipboard.

The wait starts at 0.8 seconds and can be set anywhere from 0.2 to 5; anything
outside that, or not a number at all, is corrected in the field so it never
disagrees with the timer. What the wait trades against is two-stroke digits —
4, 5, a crossed 7 — where the pointer has to reach the second stroke before the
timer runs out; miss it and a "4" is filed as "6" followed by "1", inserting a
wrong digit *and* splitting the intended one. There is more room than the number
suggests, because the 150 ms prediction debounce is spent first: at the default,
a measured 900 ms between strokes still comes out as one digit, and 1000 ms is
where a 4 breaks. Showing the countdown is what makes the rest of it work — you
can see the add coming and keep drawing, which cancels it.

The app bundle records the absolute path of both this folder and the
interpreter, because Finder launches an app with almost nothing on its PATH.
Moving the whole folder is fine — the bundle finds the project beside itself —
but if you move the app out on its own, or switch Python installations, rebuild
it:

```bash
python3 tools/make_launcher.py
```

Anything the app prints goes to `launcher.log`; if it fails to start it offers
to open that file for you.

Draw one digit at a time and make it large; a stroke that covers most of the
box downscales to roughly the thickness the network was trained on.

## Write in a browser

`web/` is a static page with no build step and no dependencies — open it through
any web server and the network runs locally in the tab:

```bash
python3 -m http.server 8000 --directory web
```

The page needs a server rather than a `file://` URL only because ES modules and
`fetch` require an origin. Nothing is uploaded; the 852 KB of weights are the
whole download.

The browser version takes a whole number rather than one digit: write across the
strip and each digit is read and collected as you move on to the next, its ink
fading once it has been taken. The last digit, with nothing after it to settle
it, is taken after the same pause, which is a field you can type into.

Digits are told apart by the gaps between them, not by time. Time alone cannot
do it — write quickly and two digits fall inside one window, write slowly and a
two-stroke digit is cut in half — whereas the strokes *within* a digit overlap
horizontally, since the bar of a 4 crosses its diagonal and the top of a 5 sits
over its curve. `segmentStrokes` in `digit-model.js` splits on horizontal gaps
and so never cuts a digit in half; `tests/test_web.mjs` writes 2026, 415 and
73908 across a strip and checks they read back in order.

Writing works with a finger as well as a mouse, the strip keeps its height and
takes the width of the page rather than being squashed on a phone, the page
follows the reader's light or dark theme, and both the pause length and whether
it applies at all are remembered between visits.

The desktop app is still one digit at a time on a square pad; the strip only
makes sense with the width to write across.

Re-export the weights after retraining:

```bash
python3 tools/export_web_model.py
```

## Classify image files

```bash
python3 tools/export_samples.py --count 30      # optional: make some test images
python3 -m digit_recognizer.predict samples/*.png
```

```
digit_00000_label7.png       -> 7      7  99.9%  9   0.0%  3   0.0%
digit_00001_label2.png       -> 2      2 100.0%  8   0.0%  7   0.0%
```

Photos and scans work as well as exported files: dark ink on light paper is
detected and inverted automatically.

## Check it still works

```bash
python3 -m unittest discover -s tests -v   # normalisation rules, no model needed
node --test tests/test_web.mjs             # the browser port against the Python one
python3 tools/simulate_drawing.py          # opens the GUI and writes 0-9 on it
```

The three suites overlap on purpose. `test_web.mjs` replays the very same
strokes `simulate_drawing.py` uses — they are stored once, in
`tests/fixtures/strokes.json` — so "the desktop app reads a 6" and "the web page
reads a 6" are the same claim about the same input.

## How it works

The architecture is unremarkable — two blocks of 3x3 convolutions with batch
norm, then a 256-unit dense layer, about 870k parameters. Two other things
matter more in practice:

**Preprocessing.** MNIST images are not plain 28x28 crops. Each digit was scaled
to fit a 20x20 box and then translated so its *centre of mass* sat at the centre
of a 28x28 frame. Feed the network a raw crop instead and accuracy collapses,
even though the picture looks fine to a human. `preprocess.py` reproduces that
normalisation exactly, and `tests/test_preprocess.py` pins the behaviour down.

**Augmentation.** MNIST was written with a pen on paper; the app's input is
drawn with a mouse, so it is shakier and the strokes are more uniform. Random
rotation, zoom and translation layers sit at the front of the model and only
activate during training, which is what closes most of that gap.

**The web build carries no framework.** Once each batch-norm is folded into the
layer in front of it — at inference it is only an affine map per channel — what
is left is convolution, max-pool, dense and relu, which is a few hundred lines
of plain JavaScript that finish in a couple of milliseconds. Weights are int8
with one scale per output channel; that is what keeps the download under a
megabyte, and it costs nothing measurable: 99.51% against the float model's
99.50%, with 99.99% of the 10,000 test predictions identical.

## Notes

- On macOS, python.org builds have no CA bundle configured, so downloading
  MNIST fails with `CERTIFICATE_VERIFY_FAILED`. `train.py` points `SSL_CERT_FILE`
  at `certifi`'s roots on startup unless you have already set it yourself.
- The GUI needs tkinter, which ships with the python.org installers. On a
  Homebrew Python, install it with `brew install python-tk`.
