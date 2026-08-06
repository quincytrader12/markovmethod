# Build Mamba Terminal on your own Windows PC

GitHub's cloud build is optional. You can build the app yourself on any
Windows 10/11 PC in a few minutes with the included **`build.bat`**. This
produces the exact same two programs the cloud build makes:

- **`mamba-web.exe`** — the neon web HUD (candlestick chart, 1D/1W/1M…2Y
  timeframes, execution panel, watchlist, live news, multi-account).
- **`mamba-terminal.exe`** — the text-based terminal.

---

## Step 1 — Install Python 3.12 (one time, free)

1. Go to <https://www.python.org/downloads/> and download **Python 3.12**.
2. Run the installer. On the first screen, **tick the box
   "Add Python to PATH"** (this is important), then click *Install Now*.

> Already have Python 3.10–3.12? You can skip this step.

## Step 2 — Get the code

Pick either option:

- **Easiest:** on the repository page click the green **`Code`** button →
  **Download ZIP**, then right-click the ZIP → **Extract All**.
- **Or with git:**
  ```
  git clone https://github.com/quincytrader12/markovmethod.git
  cd markovmethod
  git checkout claude/markov-hedge-fund-install-ajdttg
  ```

Make sure you're on the branch **`claude/markov-hedge-fund-install-ajdttg`**
(that's where all the latest features live). If you downloaded the ZIP, use
the branch dropdown on GitHub to select that branch *before* downloading.

## Step 3 — Double-click `build.bat`

Open the extracted folder and **double-click `build.bat`**.

A black window opens and does everything automatically:
- sets up a clean build environment,
- installs the app and the build tools,
- builds both `.exe` files.

The first run takes a few minutes (it downloads the libraries). Leave it
running until you see **`DONE.`**

> If Windows SmartScreen shows "Windows protected your PC", click
> **More info → Run anyway** — that warning appears for any new local script.

## Step 4 — Run the app

Your programs are in the new **`dist`** folder:

- **`dist\mamba-web.exe`** — double-click it. A console window opens, prints a
  local address like `http://127.0.0.1:8000`, and your browser opens the neon
  HUD automatically. Keep the console window open while you use the app; close
  it to stop the server.
- **`dist\mamba-terminal.exe`** — the text terminal version.

You can copy `mamba-web.exe` anywhere (Desktop, USB stick) and it runs on its
own — no Python needed to *run* it, only to build it.

---

## Connecting your Alpaca account (optional)

The app runs fully offline in demo mode. To use your own portfolios, open the
**web HUD → Accounts** panel and add your Alpaca API key/secret (paper or
live). You can save several named accounts and switch between them at runtime.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Python was not found" | Install Python 3.12 and **tick "Add Python to PATH"**, then re-run `build.bat`. |
| Install step fails on a library | Re-run `build.bat` (it resumes); make sure you have an internet connection for the first build. |
| Want a totally clean rebuild | Delete the `.buildenv`, `build`, and `dist` folders, then run `build.bat` again. |
| Antivirus flags the fresh `.exe` | Common for brand-new unsigned PyInstaller apps; allow/whitelist it, or add a code-signing certificate later. |

## Prefer a cloud build with a real installer? (AppVeyor)

If you'd rather not build locally, **AppVeyor** is a free cloud CI (separate
from GitHub Actions) that builds the apps *and* a proper installer, then gives
you download links — no Python or Inno Setup needed on your PC. The repo
already includes the config (`appveyor.yml` + `installer/mamba.iss`).

1. Go to <https://ci.appveyor.com/signup> and click **Sign in with GitHub**
   (free "Individual" plan; the repo is public so no card needed).
2. Click **New Project** → find **quincytrader12/markovmethod** → **Add**.
3. Click **New Build** (or just push a commit). AppVeyor spins up a Windows
   machine, builds everything, and runs Inno Setup.
4. When it's green, open the build → **Artifacts** tab and download:
   - **`MambaTerminalSetup.exe`** — the installer (double-click → Next → Finish;
     adds Start Menu + desktop shortcuts and an uninstaller).
   - `mamba-web.exe` / `mamba-terminal.exe` — the raw executables, if you want
     them separately.

Every future push to the branch rebuilds these automatically.

### Building the installer locally instead

If you're building locally (Steps 1–3 above) and also want the installer, just
install **Inno Setup 6** (free, <https://jrsoftware.org/isdl.php>), then run:

```
build.bat
iscc installer\mamba.iss
```

The installer appears at `installer_output\MambaTerminalSetup.exe`.

## Just want to try it without building an .exe?

If you only want to *use* the web HUD (not distribute an installer), you don't
need `build.bat` at all — after Step 1 and Step 2, open a terminal in the
folder and run:

```
py -m pip install ".[web]"
py -m markov_hedge_fund_method.web
```

That launches the same neon HUD in your browser.
