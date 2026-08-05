# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the markov-terminal Windows executable.

Built in CI (`.github/workflows/build-windows.yml`) on a windows-latest
runner:  pyinstaller --clean --noconfirm packaging/markov-terminal.spec

The tricky dependencies (textual ships .tcss data files; alpaca / yfinance /
keyring import submodules dynamically) are pulled in wholesale via
collect_all / collect_submodules so lazy imports resolve inside the bundle.
Only the terminal's import graph is packaged — proof.py/matplotlib are not
reachable from the entry point, so the exe stays lean.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is the absolute directory of this spec file (…/packaging).
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ENTRY = os.path.join(SPECPATH, "entry_terminal.py")

datas, binaries, hiddenimports = [], [], []

# Whole-package collection for the awkward ones.
for pkg in ("textual", "alpaca", "yfinance", "curl_cffi", "keyring"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Package not present at build time (e.g. curl_cffi optional) — skip.
        pass

# All of our own modules, so terminal's lazy `from .tui import ...` resolves.
hiddenimports += collect_submodules("markov_hedge_fund_method")

# keyring's Windows credential backend is imported by name at runtime.
hiddenimports += ["keyring.backends.Windows", "win32ctypes.core"]


a = Analysis(
    [ENTRY],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter", "PyQt5", "PySide6", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="markov-terminal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # TUI needs a console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
