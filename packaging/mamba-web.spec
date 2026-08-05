# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the mamba-web Windows executable (the neon web HUD).

Built in CI (`.github/workflows/build-windows.yml`) on a windows-latest
runner:  pyinstaller --clean --noconfirm packaging/mamba-web.spec

Bundles the FastAPI/uvicorn server, the alpaca / keyring stack, and the static
HUD page (web_static/index.html). uvicorn imports its protocol/loop/lifespan
implementations by name at runtime, so those submodules are collected
explicitly. The exe starts the local server and opens the browser.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ENTRY = os.path.join(SPECPATH, "entry_web.py")

datas, binaries, hiddenimports = [], [], []

# Whole-package collection for the awkward ones (data files + dynamic imports).
for pkg in ("fastapi", "starlette", "uvicorn", "pydantic", "pydantic_core",
            "anyio", "click", "h11", "sniffio", "certifi",
            "alpaca", "yfinance", "curl_cffi", "keyring"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass  # optional at build time — skip if absent

# uvicorn resolves these by name at runtime ("auto" pickers).
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on", "uvicorn.lifespan.off",
]

# Our own modules so lazy `from .web import ...` etc. resolve.
hiddenimports += collect_submodules("markov_hedge_fund_method")

# keyring's Windows credential backend is imported by name at runtime.
hiddenimports += ["keyring.backends.Windows", "win32ctypes.core"]

# The HUD page must ride along, at markov_hedge_fund_method/web_static/.
datas += [(os.path.join(REPO_ROOT, "markov_hedge_fund_method", "web_static", "index.html"),
           "markov_hedge_fund_method/web_static")]


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
    name="mamba-web",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # prints the local URL + server logs
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
