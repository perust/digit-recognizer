"""Smoke-test the GUI by writing each digit on it with synthetic mouse strokes.

Real handwriting needs a hand, so this feeds the app the same press/drag events
Tk would deliver and then reads the prediction back out of the widgets.  It
exercises the whole chain -- strokes, preprocessing, network, display -- which
the unit tests deliberately stop short of.

    python3 tools/simulate_drawing.py
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from digit_recognizer.app import DigitRecognizerApp  # noqa: E402

# The paths live in a fixture rather than in this file because the browser
# build's test suite replays the very same strokes; sharing the data is what
# makes "the desktop app reads a 6" and "the web page reads a 6" comparable.
FIXTURE = PROJECT_DIR / "tests" / "fixtures" / "strokes.json"
_fixture = json.loads(FIXTURE.read_text())

STEP = _fixture["step"]  # pixels between consecutive drag events
# One entry per digit; each entry is a list of pen-down strokes.
STROKES = {int(digit): [[tuple(point) for point in stroke] for stroke in strokes]
           for digit, strokes in _fixture["strokes"].items()}


def motion(x: float, y: float):
    """The handlers only read .x and .y, so a stand-in for tk.Event is enough."""
    return types.SimpleNamespace(x=int(round(x)), y=int(round(y)))


def write(app: DigitRecognizerApp, strokes: list) -> None:
    """Replay one digit as press / drag events, interpolated like a real pen."""
    for stroke in strokes:
        app._on_press(motion(*stroke[0]))
        for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
            count = max(1, int(math.hypot(x1 - x0, y1 - y0) / STEP))
            for i in range(1, count + 1):
                app._on_drag(motion(x0 + (x1 - x0) * i / count, y0 + (y1 - y0) * i / count))


def main() -> None:
    app = DigitRecognizerApp()
    app.update()

    correct = 0
    for expected, strokes in sorted(STROKES.items()):
        write(app, strokes)
        app._predict()  # the debounce timer never fires without a mainloop
        app.update()
        shown = app.digit_label.cget("text")
        correct += shown == str(expected)
        status = "ok" if shown == str(expected) else "MISS"
        print(f"wrote {expected} -> app shows {shown}  ({app.confidence_label.cget('text')})  {status}")
        app._commit_digit()  # appends to the field and wipes the pad for the next one

    print(f"\n{correct}/{len(STROKES)} recognised")

    # Writing the digits in order should have typed the digits in order.
    collected = app.digits.get()
    expected_text = "".join(str(digit) for digit in sorted(STROKES))
    print(f"collected field: {collected!r} (expected {expected_text!r})")

    app._copy_digits()
    pasted = app.clipboard_get()
    print(f"clipboard      : {pasted!r}")

    app.destroy()
    ok = correct == len(STROKES) and collected == expected_text and pasted == expected_text
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
