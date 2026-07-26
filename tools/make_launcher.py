"""Build "Digit Recognizer.app", a double-clickable wrapper around the board.

    python3 tools/make_launcher.py

The bundle holds no code of its own.  It draws an icon, records where the
project and the interpreter live -- neither of which is on the PATH Finder
hands to an app -- and shells out to `python3 -m digit_recognizer.app`.  Rerun
this after moving the project somewhere else.
"""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "Digit Recognizer"
EXECUTABLE_NAME = "DigitRecognizer"

ICON_SIZE = 1024
SUPERSAMPLE = 2  # draw large and shrink, so the curves come out clean
ICON_INSET = 100  # macOS leaves a margin around the rounded square
ICON_RADIUS = 185
TOP_COLOR, BOTTOM_COLOR = (44, 49, 60), (22, 24, 28)

# The "3" from the stroke simulator, in the 300x300 coordinates of the canvas.
DIGIT_STROKE = [
    (100, 82), (150, 63), (192, 88), (170, 130), (136, 142),
    (178, 152), (202, 188), (176, 228), (120, 238), (95, 216),
]
DIGIT_HEIGHT = 500  # how tall the glyph should sit inside the icon
PEN = 84

LAUNCHER_SCRIPT = r"""#!/bin/bash
# Launcher for the Handwritten Digit Recognizer.
#
# Finder starts an app with a bare PATH and an arbitrary working directory, so
# the project folder and the interpreter are both located explicitly.  Each is
# resolved relative to the bundle first, so the project can be moved as a whole,
# and falls back to the absolute path recorded when the bundle was built.

set -u

BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="$(dirname "$BUNDLE")"
[ -d "$PROJECT/digit_recognizer" ] || PROJECT="@@PROJECT@@"

# Preferred first: the interpreter symlinked into the bundle.  Running it from
# inside Contents/MacOS is what makes the Dock show this app's name and icon
# rather than a generic Python process.
PYTHON="$BUNDLE/Contents/MacOS/@@INTERPRETER@@"
[ -x "$PYTHON" ] || PYTHON="@@PYTHON@@"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"

LOG="$PROJECT/launcher.log"

alert() {
    /usr/bin/osascript -e "display dialog \"$1\" with title \"@@APPNAME@@\" \
        buttons {\"OK\"} default button \"OK\" with icon caution" >/dev/null 2>&1
}

if [ ! -d "$PROJECT/digit_recognizer" ]; then
    alert "Cannot find the project folder.\n\nKeep this app next to the digit_recognizer folder, or rebuild it with:\n    python3 tools/make_launcher.py"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    alert "Cannot find a python3 interpreter.\n\nRebuild the app with:\n    python3 tools/make_launcher.py"
    exit 1
fi

# Nothing to recognise with until the network has been trained once.
if [ ! -f "$PROJECT/models/digit_cnn.keras" ]; then
    answer=$(/usr/bin/osascript -e 'display dialog "No trained model found.\n\nTraining runs in Terminal and takes about 13 minutes." with title "@@APPNAME@@" buttons {"Cancel", "Train Now"} default button "Train Now"' 2>/dev/null)
    case "$answer" in
        *"Train Now"*)
            /usr/bin/osascript \
                -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && '$PYTHON' -m digit_recognizer.train\"" \
                -e 'tell application "Terminal" to activate' >/dev/null 2>&1
            ;;
    esac
    exit 0
fi

cd "$PROJECT" || exit 1

# Not exec'd on purpose: staying alive lets a crash be reported to the user,
# who otherwise just sees the Dock icon blink and disappear.
if ! "$PYTHON" -m digit_recognizer.app >"$LOG" 2>&1; then
    choice=$(/usr/bin/osascript -e "display dialog \"The app stopped with an error.\n\nDetails were written to launcher.log.\" with title \"@@APPNAME@@\" buttons {\"Show Log\", \"OK\"} default button \"Show Log\" with icon stop" 2>/dev/null)
    case "$choice" in *"Show Log"*) /usr/bin/open -e "$LOG" ;; esac
    exit 1
fi
"""


def render_icon() -> Image.Image:
    """Draw the app icon: a handwritten 3 on the dark rounded square of the canvas."""
    size = ICON_SIZE * SUPERSAMPLE
    scale = SUPERSAMPLE

    ramp = np.linspace(TOP_COLOR, BOTTOM_COLOR, size, dtype=np.uint8)
    background = Image.fromarray(np.repeat(ramp[:, np.newaxis, :], size, axis=1), "RGB")

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [ICON_INSET * scale, ICON_INSET * scale, size - ICON_INSET * scale, size - ICON_INSET * scale],
        radius=ICON_RADIUS * scale,
        fill=255,
    )

    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(background, (0, 0), mask)

    pen = ImageDraw.Draw(icon)
    points = _fit_stroke(DIGIT_STROKE, centre=size / 2, height=DIGIT_HEIGHT * scale)
    width = PEN * scale
    pen.line(points, fill=(255, 255, 255, 255), width=int(width), joint="curve")
    for x, y in (points[0], points[-1]):  # PIL draws butt caps; round them off
        pen.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=(255, 255, 255, 255))

    return icon.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)


def _fit_stroke(stroke: list, centre: float, height: float) -> list:
    """Scale a canvas-space stroke to `height` pixels and centre it on the icon."""
    xs, ys = [p[0] for p in stroke], [p[1] for p in stroke]
    factor = height / (max(ys) - min(ys))
    mid_x, mid_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return [(centre + (x - mid_x) * factor, centre + (y - mid_y) * factor) for x, y in stroke]


def write_icns(icon: Image.Image, destination: Path, work_dir: Path) -> None:
    """Convert the artwork into the multi-resolution .icns format Finder wants."""
    iconset = work_dir / "icon.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir(parents=True)

    for base in (16, 32, 128, 256, 512):
        for suffix, pixels in ((f"{base}x{base}", base), (f"{base}x{base}@2x", base * 2)):
            icon.resize((pixels, pixels), Image.LANCZOS).save(iconset / f"icon_{suffix}.png")

    subprocess.run(
        ["iconutil", "--convert", "icns", str(iconset), "--output", str(destination)], check=True
    )
    shutil.rmtree(iconset, ignore_errors=True)


def write_plist(destination: Path) -> None:
    destination.write_bytes(
        plistlib.dumps(
            {
                "CFBundleName": APP_NAME,
                "CFBundleDisplayName": APP_NAME,
                "CFBundleExecutable": EXECUTABLE_NAME,
                "CFBundleIconFile": "icon",
                "CFBundleIdentifier": "local.digit-recognizer",
                "CFBundleInfoDictionaryVersion": "6.0",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "1.0",
                "CFBundleVersion": "1",
                "LSMinimumSystemVersion": "11.0",
                "NSHighResolutionCapable": True,
            }
        )
    )


def gui_interpreter(python: Path) -> Path:
    """The interpreter binary that is allowed to own a GUI process.

    What python.org installs in `bin/` is a wrapper that re-executes itself out
    of the framework's own Python.app the moment Tk starts up.  That hands the
    running process *that* bundle's identity instead of ours, so link straight
    to the binary it would have jumped to.  Other builds -- Homebrew, pyenv --
    have no such wrapper and are used as they are.
    """
    bundled = Path(sys.prefix) / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return bundled if bundled.exists() else python


def link_interpreter(macos_dir: Path, python: Path) -> Path:
    """Symlink the interpreter into the bundle, under the name of the app.

    Two separate things follow from where a running process's executable file
    sits.  Because it is inside Contents/MacOS, CoreFoundation treats this
    bundle as the main one and the Dock picks up the icon drawn above; and
    because macOS names a process after that file, calling the link "Digit
    Recognizer" is what stops the Dock and menu bar from saying "python3".
    """
    link = macos_dir / APP_NAME
    link.unlink(missing_ok=True)
    link.symlink_to(gui_interpreter(python))
    return link


def build(output_dir: Path) -> Path:
    if sys.platform != "darwin":
        raise SystemExit("this builder makes a macOS .app bundle; it only runs on macOS")

    bundle = output_dir / f"{APP_NAME}.app"
    shutil.rmtree(bundle, ignore_errors=True)
    contents = bundle / "Contents"
    macos_dir, resources = contents / "MacOS", contents / "Resources"
    macos_dir.mkdir(parents=True)
    resources.mkdir(parents=True)

    python = Path(sys.executable).resolve()
    write_plist(contents / "Info.plist")
    write_icns(render_icon(), resources / "icon.icns", work_dir=contents)
    link_interpreter(macos_dir, python)

    launcher = macos_dir / EXECUTABLE_NAME
    launcher.write_text(
        LAUNCHER_SCRIPT.replace("@@PROJECT@@", str(PROJECT_DIR))
        .replace("@@INTERPRETER@@", APP_NAME)
        .replace("@@PYTHON@@", str(gui_interpreter(python)))
        .replace("@@APPNAME@@", APP_NAME)
    )
    launcher.chmod(0o755)

    # Finder caches icons per bundle; touching it makes the new one show up.
    subprocess.run(["touch", str(bundle)], check=False)

    print(f"built {bundle}")
    print(f"  project     {PROJECT_DIR}")
    print(f"  interpreter {gui_interpreter(python)}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a double-clickable macOS launcher.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
