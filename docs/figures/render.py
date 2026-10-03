"""Render the README figures (docs/figures/*.html) to docs/images/*.png with headless Edge/Chrome.

Usage (from the project root, after `npm install` in web/ so the bundled fonts exist):
    .venv\\Scripts\\python.exe docs/figures/render.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGURES = {"pipeline": 1600, "architecture": 1600, "results": 1600}
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    found = shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")
    if not found:
        sys.exit("no Edge / Chrome found")
    return found


def trim_bottom(path: Path, margin: int = 44) -> None:
    """Cut the empty paper below the figure, keeping the page's bottom margin."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    bg = a[-1, 0]
    rows = np.where((np.abs(a.astype(int) - bg.astype(int)).sum(axis=2) > 12).any(axis=1))[0]
    bottom = min(a.shape[0], int(rows[-1]) + 1 + margin) if len(rows) else a.shape[0]
    im.crop((0, 0, a.shape[1], bottom)).save(path, optimize=True)


def main() -> None:
    exe = browser()
    out_dir = ROOT / "docs" / "images"
    with tempfile.TemporaryDirectory() as profile:
        for name, width in FIGURES.items():
            src = ROOT / "docs" / "figures" / f"{name}.html"
            out = out_dir / f"{name}.png"
            subprocess.run(
                [
                    exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--allow-file-access-from-files", f"--user-data-dir={profile}",
                    f"--window-size={width},1500", f"--screenshot={out}", src.as_uri(),
                ],
                check=True, capture_output=True, timeout=120,
            )  # fmt: skip
            trim_bottom(out)
            w, h = Image.open(out).size
            print(f"{out.relative_to(ROOT)}  {w}x{h}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
