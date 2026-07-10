# Chess.com Assistant ♟️👁️🤖

**A computer-vision + Stockfish assistant that reads the board on Chess.com, finds strong moves in real time, and can play them for you through the browser — with human-like timing and anti-detection built in.**

> ⚠️ **Educational / research project.** Using engine assistance in live games violates the Chess.com Fair Play policy and can get your account banned. Use it on your own account, against the bots, at your own risk.

## Features ✨
- 👁️ **Board detection (OpenCV)** — finds the board on screen by its colors and verifies the checkerboard pattern. Auto-locates the board at any resolution and auto-detects which color you're playing (from the board plus Chess.com's DOM). Resolution-independent, no machine-learning model required. *(The old PyTorch/ResNet50 recognizer has been replaced with fast template matching.)*
- 🤖 **Stockfish engine** — skill levels 0–20, adaptive search depth with automatic deepening on critical positions.
- ♟️ **Real game-state tracking (python-chess)** — reconstructs the full position (castling, en-passant, move counters) by matching each detected board to a legal move, with a color-only fallback and FEN re-sync when tracking drifts.
- 🖱️ **Auto-play via Selenium** — clicks moves directly into the Chess.com DOM, so it works on **Wayland** (not just X11). Human-like mouse paths: Bézier curves, ease-in/out, slight overshoot and tremor. Falls back to pyautogui on X11.
- 🕵️ **Anti-detection**
  - Bimodal, clock-aware "human think time"
  - Randomized depth, exponential move-weighting, occasional lazy/blunder moves
  - Dynamic skill adjustment that estimates the opponent's strength (from their centipawn loss) and plays *slightly* above them
  - Built-in opening book for the first several moves
- 🔐 **Session persistence** — you log in by hand once in the Selenium browser (solving Chess.com's human-check yourself); it saves that session and reuses it, so later runs start already logged in and skip the login page.
- 🔄 **Signed self-update** — checks GitHub at startup and, if a newer release exists, verifies its Ed25519 signature against a key baked into the app before offering to install it.
- 📊 **Opponent analysis** — tracks the opponent's average centipawn loss and flags likely engine users.
- 🎮 **Modes & marathon**
  - Play vs **bots**, vs **online players**, or **watch** (suggest-only)
  - **Marathon**: auto-queues new games, with a live W/L/D record and streaks
  - Hotkeys: **P** pause/resume, **L** finish the current game then stop
  - `-r` resumes your last session
- 🛟 Game-over detection (DOM + visual) and automatic restart if the browser crashes.

## How It Works 🔧
1. **Vision** — `pyscreenshot` grabs the screen → OpenCV locates the board by HSV color + checkerboard validation → each cell is classified (empty / white / black) and identified by per-cell template matching.
2. **State** — detected boards are matched against python-chess legal moves to keep an exact `GameState`; a FEN re-sync recovers if tracking drifts.
3. **Move choice** — opening book first, otherwise Stockfish `multipv` analysis with weighted, human-like selection and optional blunders.
4. **Play** — Selenium moves the mouse along a natural curve and clicks the from/to squares after a realistic think-time delay.

## Requirements 📦
```
Python 3.8+
Stockfish                 # engine
Firefox + geckodriver     # for Selenium auto-play
python-chess
opencv-python
numpy
pyscreenshot
selenium
pyautogui                 # X11 fallback
psutil
pillow
cryptography              # verifies signed self-updates
```
> No `torch` / `torchvision` anymore — the vision pipeline is pure OpenCV now.

## Installation 🛠️
```bash
pip install -r requirements.txt
# (on newer distros you may need:  pip install --break-system-packages -r requirements.txt )

# Engine + browser
sudo apt-get install stockfish firefox     # Debian/Ubuntu
sudo snap install geckodriver              # or grab it from the Mozilla releases page
# macOS:  brew install stockfish
```
**Windows:** download Stockfish from https://stockfishchess.org/download/ and geckodriver from the Mozilla `geckodriver` releases page.

### Or let it install itself 🪄
On first run, if anything's missing the app **offers to set it up for you** — Python packages via `pip` (PyPI), plus **Stockfish** and **geckodriver** as **SHA256-verified official downloads** (from `github.com/official-stockfish` and `github.com/mozilla`) into `~/.config/chessassistant/bin/`. No admin/sudo, and it only ever pulls from those official hosts. Run it up front anytime with:
```bash
python3 chessAssistant.py --setup
```
(Firefox itself isn't auto-installed — it just points you to mozilla.org if it's missing.)

## Usage ▶️
```bash
python3 chessAssistant.py        # then answer the prompts
python3 chessAssistant.py -r     # resume last session
```
You'll be asked for the skill level, whether to use the anti-detection randomizer, whether to auto-move, the mode (bots / players / watch), the time control (a standard one like `15|10` or a custom `M|inc` / days), and marathon on/off. Settings are saved to `config.json` and `last_session.json`.

## Notes 🗒️
- **Ideal layout:** Chess.com on the **left half** of the screen, terminal on the **right**. Works best at **1440p** on the **default Chess.com theme**.
- Auto-play opens its own Selenium-controlled Firefox window, so it works on Wayland; the pyautogui path needs X11.
- Windows is supported (screen capture falls back to Pillow's `ImageGrab`).
- Files it writes next to the script: `current_fen.txt` (the live position), `annotated_chessboard.png` (a debug overlay of what it detected), `chess_engine_error.log`, plus your saved `config.json` / `last_session.json`.
- *Work in progress — expect rough edges.* :p

## Screenshots 📸
![chessAssistant on Linux](https://github.com/thetrueartist/chess.comAssistant/blob/main/chessAssistant.png)
![chessAssistant on Windows](https://github.com/thetrueartist/chess.comAssistant/blob/main/chessAssistantWINDOWS.png)
