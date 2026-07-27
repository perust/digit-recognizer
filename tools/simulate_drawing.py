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

from digit_recognizer.app import DEFAULT_AUTO_COMMIT_MS, DigitRecognizerApp  # noqa: E402

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

    automatic = check_auto_commit(app)
    print(f"pen left resting -> collected {automatic!r} with no key pressed")

    typed = check_typed_pause(app)
    print(f"pause typed as {typed['typed']!r} -> honoured as {typed['accepted']!r}, "
          f"nonsense falls back to {typed['fallback']!r}")

    app.destroy()
    ok = (
        correct == len(STROKES)
        and collected == expected_text
        and pasted == expected_text
        and automatic == "5"
        and typed["accepted"] == "2.0"
        and typed["fallback"] == f"{DEFAULT_AUTO_COMMIT_MS / 1000:.1f}"
        and typed["clamped"] == "5.0"
    )
    sys.exit(0 if ok else 1)


def check_auto_commit(app: DigitRecognizerApp, pause: str | None = None) -> str:
    """Write a 5, drop the pen, and let the timers run without touching a key.

    Needs a real event loop: the countdown and the prediction debounce are both
    `after` callbacks, which never fire while the rest of this script drives the
    app synchronously.
    """
    if pause is not None:
        app.pause_seconds.set(pause)
    app.digits.set("")
    write(app, STROKES[5])
    app._on_release(None)  # pen up is what arms the countdown
    app.after(app._pause_ms() + 900, app.quit)
    app.mainloop()
    return app.digits.get()


def check_typed_pause(app: DigitRecognizerApp) -> dict:
    """A typed wait should be taken as written, or corrected if it cannot be."""
    results = {}
    for key, typed in (("accepted", "2"), ("fallback", "abc"), ("clamped", "99")):
        app.pause_seconds.set(typed)
        app._normalise_pause()
        results[key] = app.pause_seconds.get()
    results["typed"] = "2"

    app.pause_seconds.set("0.3")
    app._normalise_pause()
    assert check_auto_commit(app) == "5", "a shortened wait should still collect the digit"
    return results


if __name__ == "__main__":
    main()
