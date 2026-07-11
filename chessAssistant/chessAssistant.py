import os
import platform
import sys

# ── Dependency bootstrap ──────────────────────────────────────────────────────
# Runs BEFORE the third-party imports below. If the Python packages the app needs
# aren't installed, offer to install them from PyPI via pip (official + hash-
# verified by pip). Opt-in, no sudo. Binaries (Stockfish/geckodriver) are handled
# on demand later by ensure_stockfish()/ensure_geckodriver(). Skip with the env
# var CHESSASSIST_NO_BOOTSTRAP=1.
def _bootstrap_python_deps():
    deps = {  # import-name : pip-name
        "cv2": "opencv-python", "numpy": "numpy", "chess": "python-chess",
        "selenium": "selenium", "psutil": "psutil", "PIL": "pillow",
        "cryptography": "cryptography", "pyautogui": "pyautogui",
    }
    if platform.system() != "Windows":
        deps["pyscreenshot"] = "pyscreenshot"
    import importlib.util

    def have(mod):
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    missing = sorted({pip for mod, pip in deps.items() if not have(mod)})
    if not missing:
        return
    print("\033[93mMissing Python packages:\033[0m " + ", ".join(missing))
    try:
        ans = input("Install them now from PyPI (pip)? [Y/n]: ").strip().lower()
    except EOFError:
        ans = "n"
    if ans not in ("", "y", "yes"):
        print("Skipping — the app may not start (pip install " + " ".join(missing) + ").")
        return
    import subprocess
    base = [sys.executable, "-m", "pip", "install"]
    for extra in ([], ["--user"], ["--break-system-packages"]):
        try:
            subprocess.check_call(base + extra + missing)
            break
        except Exception:
            continue
    if any(not have(mod) for mod, pip in deps.items() if pip in missing):
        print("\033[91mCould not install some packages\033[0m — install manually and re-run.")
        sys.exit(1)
    # Re-launch so the freshly installed packages import cleanly (POSIX). On
    # Windows, execv garbles the console, so ask for a manual restart instead.
    if platform.system() == "Windows" or os.environ.get("_CHESSASSIST_REEXEC") == "1":
        print("\033[92mDependencies installed — please run the script again.\033[0m")
        sys.exit(0)
    os.environ["_CHESSASSIST_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if os.environ.get("CHESSASSIST_NO_BOOTSTRAP") != "1":
    _bootstrap_python_deps()

import cv2
import numpy as np
if platform.system() == "Windows":
    from PIL import ImageGrab
else:
    import pyscreenshot as ImageGrab
import chess
import chess.engine
import threading
import time
import signal
import psutil
import queue
import logging
import random
import tempfile
import json

# Configure logging
logging.basicConfig(
    filename="chess_engine_error.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# When Firefox dies, Selenium/urllib3 flood the log with connection-retry WARNINGs
# (100+ per crash). We detect and recover from a dead session ourselves, so silence
# that noise and keep only real errors.
for _noisy in ("urllib3", "urllib3.connectionpool", "selenium",
               "selenium.webdriver.remote.remote_connection"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Detection can misread pieces into a physically-impossible board (two kings, three
# queens, pawns on the back rank, ...). Syncing to those corrupts the game state and
# hangs the bot (Stockfish can't analyse a 2-king board → it can't move → flags on
# time). Both sync paths reject any position carrying one of these structural faults.
_ILLEGAL_STATUS = (chess.Status.TOO_MANY_KINGS | chess.Status.NO_WHITE_KING
                   | chess.Status.NO_BLACK_KING | chess.Status.TOO_MANY_WHITE_PIECES
                   | chess.Status.TOO_MANY_BLACK_PIECES | chess.Status.TOO_MANY_WHITE_PAWNS
                   | chess.Status.TOO_MANY_BLACK_PAWNS | chess.Status.PAWNS_ON_BACKRANK)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# Default configuration
DEFAULT_CONFIG = {
    "threads": max(4, os.cpu_count() // 2),
    "hash_size": 4096,
    "max_depth": 30,
    "template_match_scales": [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    "piece_detect_threshold": 0.12,
    "board_redetect_interval": 30,  # Re-detect board location every N cycles
    "move_stabilization_attempts": 5,
    "move_stabilization_delay": 0.2,
    "loop_delay": 0.4,
}


def load_config():
    """Load config from file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
    return config


def save_config(config):
    """Save current config to file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


SESSION_PATH = os.path.join(SCRIPT_DIR, "last_session.json")

def save_session(skill_level, use_randomizer, auto_move, game_mode, marathon,
                 time_control=None, auto_report=False):
    """Save session settings for -r resume."""
    with open(SESSION_PATH, "w") as f:
        json.dump({
            "skill_level": skill_level,
            "use_randomizer": use_randomizer,
            "auto_move": auto_move,
            "game_mode": game_mode,
            "marathon": marathon,
            "time_control": time_control,
            "auto_report": auto_report,
        }, f, indent=2)


def load_session():
    """Load last session settings. Returns dict or None."""
    if os.path.exists(SESSION_PATH):
        try:
            with open(SESSION_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ─── Auto-Update (check + notify only, Ed25519 signature-verified) ────────────
# NOTE: this NEVER downloads or overwrites any file — it only tells you when a
# newer, cryptographically-signed release exists on GitHub. Applying it is manual.

__version__ = "6.0.14"  # bump on each release; the updater compares this to GitHub
RELEASE_SIGNING_PUBKEY_B64 = "wtPazhR1+uBdRVNqjxZut4EbnKMzdWlfkmk+BURy9R8="
_UPDATE_RAW_BASE = ("https://raw.githubusercontent.com/thetrueartist/"
                    "chess.comAssistant/main/chessAssistant")


def _parse_version(s):
    try:
        return tuple(int(x) for x in str(s).strip().split("."))
    except Exception:
        return None


def _verify_release_signature(data_bytes, sig_b64):
    """Verify an Ed25519 signature over data_bytes against the embedded public key.
    Returns True (valid), False (invalid/mismatch), or None (crypto unavailable)."""
    try:
        import base64
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except Exception:
        return None
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(RELEASE_SIGNING_PUBKEY_B64))
        pub.verify(base64.b64decode((sig_b64 or "").strip()), data_bytes)
        return True
    except Exception:
        return False


def apply_update(remote_bytes, remote_ver):
    """Overwrite this file with the (already signature-verified) new version,
    keeping a .bak, then relaunch. Only call AFTER the signature has verified."""
    import sys, shutil as _sh
    target = os.path.realpath(__file__)
    try:
        _sh.copy2(target, target + ".bak")
        with open(target, "wb") as f:
            f.write(remote_bytes)
        print(f"\033[92m✓ Updated to v{remote_ver}  (backup: {os.path.basename(target)}.bak)\033[0m")
        if platform.system() == "Windows":
            # os.execv doesn't cleanly replace the process on Windows — it returns
            # to the shell and garbles the console — so exit cleanly and let the
            # user restart instead of auto-relaunching.
            print("\033[92m  Update installed. Restart to use it:\033[0m python chessAssistant.py")
            os._exit(0)
        print("\033[90m  Restarting...\033[0m")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        logging.error(f"apply_update failed: {e}")
        print(f"\033[91mUpdate failed ({e}) — current version unchanged. "
              f"Update manually with: git pull\033[0m")


def check_for_update(timeout=4):
    """Check GitHub for a newer, signature-verified release. If one is found and
    its signature verifies, offer to install it (backs up + relaunches).
    Safe to call at startup; never raises."""
    import urllib.request, re

    def _fetch(url):
        req = urllib.request.Request(url, headers={"User-Agent": "chessAssistant-updater"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    try:
        remote_bytes = _fetch(_UPDATE_RAW_BASE + "/chessAssistant.py")
    except Exception as e:
        logging.info(f"Update check skipped: {e}")
        return
    m = re.search(rb'^__version__\s*=\s*["\']([^"\']+)["\']', remote_bytes, re.M)
    if not m:
        return  # remote has no version marker yet — nothing to compare
    remote_ver = m.group(1).decode()
    rv, lv = _parse_version(remote_ver), _parse_version(__version__)
    if not rv or not lv or rv <= lv:
        logging.info(f"Up to date (local {__version__}, remote {remote_ver})")
        return

    # Newer version found — verify its signature BEFORE recommending it.
    try:
        sig_b64 = _fetch(_UPDATE_RAW_BASE + "/chessAssistant.py.sig").decode()
    except Exception:
        sig_b64 = None
    verdict = _verify_release_signature(remote_bytes, sig_b64) if sig_b64 else False

    print()
    if verdict is True:
        print(f"\033[92m🔔 Update available: v{remote_ver} (you have v{__version__}) — signature ✓ verified\033[0m")
        try:
            ans = input("   Install it now? (y/n): ").strip().lower()
        except Exception:
            ans = "n"
        if ans == "y":
            apply_update(remote_bytes, remote_ver)   # backs up, writes, relaunches
        else:
            print(f"\033[90m   Skipped. (You can also update later with: cd {SCRIPT_DIR} && git pull)\033[0m")
    elif verdict is None:
        print(f"\033[93m🔔 Update available: v{remote_ver} (you have v{__version__}) — signature NOT checked (no 'cryptography' installed)\033[0m")
    else:
        print(f"\033[91m⚠ A newer file (v{remote_ver}) is on GitHub but its signature did NOT verify — do NOT update; the repo/account may be compromised.\033[0m")
    print()


config = load_config()


# ─── Game State Tracker ───────────────────────────────────────────────────────

class GameState:
    """
    Tracks the full game state using python-chess as source of truth.
    Properly handles castling rights, en passant, move counters, and turn.
    """

    def __init__(self, playing_color):
        self.playing_color = playing_color
        self.board = chess.Board()
        self.move_history = []
        self.detected_boards = []  # History of raw detected boards
        self.is_our_turn = (playing_color == "w")
        self.game_active = True
        self.consecutive_same = 0
        self.last_raw_board = None

    def reset(self):
        """Reset for a new game."""
        self.board = chess.Board()
        self.move_history = []
        self.detected_boards = []
        self.is_our_turn = (self.playing_color == "w")
        self.game_active = True
        self.consecutive_same = 0
        self.last_raw_board = None

    @property
    def fen(self):
        return self.board.fen()

    @property
    def piece_count(self):
        return len(self.board.piece_map())

    @property
    def game_stage(self):
        pc = self.piece_count
        if pc > 25:
            return "opening"
        elif pc > 14:
            return "middlegame"
        return "endgame"

    def detected_board_to_piece_map(self, raw_board):
        """Convert our 8x8 detected array to a dict of {square: piece}."""
        pieces = {}
        for row in range(8):
            for col in range(8):
                p = raw_board[row][col]
                if p:
                    # Row 0 = rank 8, col 0 = file a
                    square = chess.square(col, 7 - row)
                    pieces[square] = p
        return pieces

    def _boards_match(self, raw_board):
        """Check if detected board matches current python-chess board state.
        First tries exact piece type match, then falls back to color-only
        (handles template misidentification across different piece themes)."""
        detected = self.detected_board_to_piece_map(raw_board)
        current = {sq: p.symbol() for sq, p in self.board.piece_map().items()}
        # Same occupied squares?
        if set(detected.keys()) != set(current.keys()):
            return False
        # Exact match?
        if all(detected[sq] == current[sq] for sq in detected):
            return True
        # Color-only match (same squares occupied, same colors, types may differ)
        return all(detected[sq].isupper() == current[sq].isupper() for sq in detected)

    def _color_map_from_raw(self, raw_board):
        """Convert raw board with color-only markers (W/B) to {square: color}."""
        colors = {}
        for row in range(8):
            for col in range(8):
                p = raw_board[row][col]
                if p == "W":
                    colors[chess.square(col, 7 - row)] = "w"
                elif p == "B":
                    colors[chess.square(col, 7 - row)] = "b"
                elif p:
                    # Normal piece symbol — derive color from case
                    sq = chess.square(col, 7 - row)
                    colors[sq] = "w" if p.isupper() else "b"
        return colors

    def find_matching_move_color_only(self, raw_board):
        """
        Find a legal move using only piece color information (no piece types).
        Works by checking which squares are occupied and piece colors after each legal move.
        Returns the move or None.
        """
        detected_colors = self._color_map_from_raw(raw_board)
        detected_occupied = set(detected_colors.keys())

        best_match = None
        best_score = -1
        second_best_score = -1

        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            test_colors = {}
            for sq, p in test_board.piece_map().items():
                test_colors[sq] = "w" if p.color == chess.WHITE else "b"
            test_occupied = set(test_colors.keys())

            all_squares = detected_occupied | test_occupied
            matches = sum(1 for sq in all_squares
                          if detected_colors.get(sq) == test_colors.get(sq))
            total = len(all_squares)
            score = matches / max(total, 1)

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = move
            elif score > second_best_score:
                second_best_score = score

        if best_score >= 0.95 and (best_score - second_best_score) >= 0.02:
            return best_match
        return None

    def boards_match_color_only(self, raw_board):
        """Check if detected board matches current state using only piece colors.
        Tolerates up to 1 mismatched square for robustness."""
        detected_colors = self._color_map_from_raw(raw_board)
        current_colors = {}
        for sq, p in self.board.piece_map().items():
            current_colors[sq] = "w" if p.color == chess.WHITE else "b"
        all_squares = set(list(detected_colors.keys()) + list(current_colors.keys()))
        mismatches = sum(1 for sq in all_squares
                         if detected_colors.get(sq) != current_colors.get(sq))
        return mismatches <= 1

    def find_matching_move(self, raw_board):
        """
        Given a newly detected board state, find which legal move
        transforms our current board into the detected state.
        Returns the move or None.
        """
        detected_pieces = self.detected_board_to_piece_map(raw_board)

        best_match = None
        best_score = -1
        second_best_score = -1

        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)

            test_pieces = {sq: p.symbol() for sq, p in test_board.piece_map().items()}

            # Compare all occupied squares
            all_squares = set(list(detected_pieces.keys()) + list(test_pieces.keys()))
            matches = sum(1 for sq in all_squares
                          if detected_pieces.get(sq, "") == test_pieces.get(sq, ""))
            total = len(all_squares)

            score = matches / max(total, 1)
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = move
            elif score > second_best_score:
                second_best_score = score

        # Require perfect or near-perfect match, AND clear separation from runner-up
        if best_score >= 0.97 and (best_score - second_best_score) >= 0.02:
            return best_match
        return None

    def update_from_detection(self, raw_board):
        """
        Update game state from a detected board.
        Returns: 'our_turn', 'their_turn', 'no_change', 'synced', or 'uncertain'
        """
        # Check if board changed from last detection
        if self.last_raw_board is not None:
            changed = any(raw_board[i][j] != self.last_raw_board[i][j]
                          for i in range(8) for j in range(8))
            if not changed:
                self.consecutive_same += 1
                self._pending_move = None   # board reverted — drop any unconfirmed move
                return "no_change"

        self.consecutive_same = 0

        # First check: does the detected board already match our current state?
        if self._boards_match(raw_board):
            self.last_raw_board = [row[:] for row in raw_board]
            self._pending_move = None
            if self.is_our_turn:
                return "our_turn"
            return "their_turn"

        # Try to find a matching legal move (exact piece types)
        move = self.find_matching_move(raw_board)

        # Fallback: try color-only matching if exact matching fails
        # (handles piece type misidentification across themes)
        if move is None:
            move = self.find_matching_move_color_only(raw_board)
            if move is not None:
                logging.info(f"Color-only match found: {move.uci()}")

        if move is not None:
            # Two-frame confirmation: a transient bad read (alt-tabbing, a
            # mid-animation frame, or the board briefly obscured) can look like a
            # legal move. Require the SAME move on two consecutive detections before
            # committing — fixes the "says it moved when it hasn't" tab glitch.
            if getattr(self, '_pending_move', None) != move.uci():
                self._pending_move = move.uci()
                return "no_change"   # hold: don't commit or advance last_raw_board yet
            self._pending_move = None
            self.board.push(move)
            self.move_history.append(move)
            self.last_raw_board = [row[:] for row in raw_board]
            self.is_our_turn = not self.is_our_turn
            logging.info(f"Detected move: {move.uci()}, FEN: {self.fen}")

            if self.is_our_turn:
                return "our_turn"
            return "their_turn"

        # No legal move matched — try sync from detection (with strict sanity checks)
        # Skip sync if we just auto-played (phantom pieces from move highlights)
        skip_sync = getattr(self, '_skip_sync_cycles', 0)
        if skip_sync > 0:
            self._skip_sync_cycles = skip_sync - 1
            logging.info(f"Sync skipped ({skip_sync} remaining) — waiting for opponent move")
            return "uncertain"

        fen = self._raw_board_to_fen(raw_board)
        try:
            new_board = chess.Board(fen)
            new_pieces = len(new_board.piece_map())
            old_pieces = len(self.board.piece_map())

            # Build maps for comparison
            old_map = {sq: p.symbol() for sq, p in self.board.piece_map().items()}
            new_map = {sq: p.symbol() for sq, p in new_board.piece_map().items()}
            all_sq = set(list(old_map.keys()) + list(new_map.keys()))
            diffs = sum(1 for sq in all_sq if old_map.get(sq) != new_map.get(sq))

            # Fresh-game seed: if our internal board is still the pristine starting
            # position but the detection clearly isn't, we were just (re)created mid-game
            # (stall-escape / invalid-read reset at GameState(playing_color)) and have
            # never synced to reality. The incremental-change guards below assume a valid
            # prior and would reject the real board forever ("too many pieces lost 32→N",
            # "diffs > 6"), stalling until the game abandons. There is no trustworthy prior
            # here, so seed straight from the detection as long as it is a legal position.
            if (self.board.board_fen() == chess.STARTING_BOARD_FEN
                    and new_board.board_fen() != chess.STARTING_BOARD_FEN):
                if new_board.status() & _ILLEGAL_STATUS:
                    logging.warning(f"Seed sync rejected: illegal position {fen}")
                    return "uncertain"
                self.board = new_board
                self.last_raw_board = [row[:] for row in raw_board]
                self.is_our_turn = (self.board.turn == chess.WHITE) == (self.playing_color == "w")
                logging.info(f"Seeded board from detection after reset: {fen}")
                return "our_turn" if self.is_our_turn else "synced"

            # Reject criteria
            rejected = False
            if new_pieces > 32 or new_pieces < 2:
                logging.warning(f"Sync rejected: bad piece count {new_pieces}")
                rejected = True
            elif new_pieces > old_pieces + 1:
                logging.warning(f"Sync rejected: pieces jumped {old_pieces}→{new_pieces}")
                rejected = True
            elif old_pieces - new_pieces > 3:
                logging.warning(f"Sync rejected: too many pieces lost {old_pieces}→{new_pieces}")
                rejected = True
            elif diffs > 6:
                # A legal move changes max 4 squares (castling). >6 = detection garbage.
                logging.warning(f"Sync rejected: {diffs} squares differ (max 6)")
                rejected = True
            elif new_board.status() & _ILLEGAL_STATUS:
                # Physically-impossible board (two kings, three queens, ...) — mis-detection.
                logging.warning(f"Sync rejected: illegal/mis-detected position {fen}")
                rejected = True

            if not rejected:
                # Strip phantom pieces (+1 extra)
                if new_pieces == old_pieces + 1:
                    extra = [sq for sq in new_map if sq not in old_map]
                    for sq in extra:
                        new_board.remove_piece_at(sq)
                    logging.info(f"Stripped phantom at {[chess.SQUARE_NAMES[s] for s in extra]}")

                self.board = new_board
                self.last_raw_board = [row[:] for row in raw_board]
                self.is_our_turn = (self.board.turn == chess.WHITE) == (self.playing_color == "w")
                logging.info(f"Board synced: {fen} ({diffs} diffs)")
                if self.is_our_turn:
                    return "our_turn"
                return "synced"
                self.board = new_board
                self.last_raw_board = [row[:] for row in raw_board]
                self.is_our_turn = (self.board.turn == chess.WHITE) == (self.playing_color == "w")
                logging.info(f"Board synced from detection: {fen}")
                if self.is_our_turn:
                    return "our_turn"
                return "synced"
        except ValueError:
            pass

        # Can't match - might be mid-animation or detection error
        # Don't update last_raw_board so next cycle retries matching
        logging.warning("Could not match detected board to any legal move")
        return "uncertain"

    def update_from_detection_color_only(self, raw_board):
        """
        Update game state using only piece color information (no piece types).
        Used when cells are too small for reliable template matching.
        Returns same values as update_from_detection.
        """
        # Check if board changed
        if self.last_raw_board is not None:
            changed = any(raw_board[i][j] != self.last_raw_board[i][j]
                          for i in range(8) for j in range(8))
            if not changed:
                self.consecutive_same += 1
                return "no_change"

        self.consecutive_same = 0

        # Check if colors match current state
        if self.boards_match_color_only(raw_board):
            self.last_raw_board = [row[:] for row in raw_board]
            if self.is_our_turn:
                return "our_turn"
            return "their_turn"

        # Try color-only move matching
        move = self.find_matching_move_color_only(raw_board)
        if move is not None:
            self.board.push(move)
            self.move_history.append(move)
            self.last_raw_board = [row[:] for row in raw_board]
            self.is_our_turn = not self.is_our_turn
            logging.info(f"Color-only detected move: {move.uci()}, FEN: {self.fen}")
            if self.is_our_turn:
                return "our_turn"
            return "their_turn"

        self.last_raw_board = [row[:] for row in raw_board]
        logging.warning("Color-only: could not match board to any legal move")
        return "uncertain"

    def force_sync(self, raw_board):
        """Force sync the board state from detection (used when tracking is lost).
        REJECTS physically-impossible positions (two kings, three queens, too many
        pawns, etc.) that come from a mis-detected board — syncing to those corrupts
        the game state and hangs the bot (Stockfish can't analyse a 2-king board)."""
        fen = self._raw_board_to_fen(raw_board)
        try:
            new_board = chess.Board(fen)
        except ValueError as e:
            logging.error(f"Force sync failed: {e}")
            return False
        # Legality gate: bail on detection garbage (structural king/piece/pawn faults).
        if new_board.status() & _ILLEGAL_STATUS:
            logging.warning(f"Force sync rejected: illegal/mis-detected position {fen}")
            return False
        self.board = new_board
        self.last_raw_board = [row[:] for row in raw_board]
        self.is_our_turn = (self.board.turn == chess.WHITE) == (self.playing_color == "w")
        logging.info(f"Force synced board: {fen}")
        return True

    def _raw_board_to_fen(self, raw_board):
        """Convert raw 8x8 array to FEN position string with proper metadata."""
        fen_rows = []
        for row in raw_board:
            fen_row = ""
            empty = 0
            for cell in row:
                if cell == "":
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)

        position = "/".join(fen_rows)

        # Determine turn from whose perspective we're playing
        turn = "w" if self.is_our_turn == (self.playing_color == "w") else "b"

        # Infer castling rights from piece positions
        castling = ""
        # White king on e1, rooks on a1/h1
        if raw_board[7][4] == "K":
            if raw_board[7][7] == "R":
                castling += "K"
            if raw_board[7][0] == "R":
                castling += "Q"
        if raw_board[0][4] == "k":
            if raw_board[0][7] == "r":
                castling += "k"
            if raw_board[0][0] == "r":
                castling += "q"
        if not castling:
            castling = "-"

        return f"{position} {turn} {castling} - 0 1"


# ─── Screen Capture ───────────────────────────────────────────────────────────

def capture_screenshot(bbox=None):
    """Capture screen region. If bbox is None, captures full screen."""
    temp_dir = tempfile.gettempdir()
    screenshot_path = os.path.join(temp_dir, "chessboard.png")
    if bbox:
        screenshot = ImageGrab.grab(bbox).convert("RGB")
    else:
        screenshot = ImageGrab.grab().convert("RGB")
    screenshot.save(screenshot_path)
    return screenshot_path


# ─── Board Detection ──────────────────────────────────────────────────────────

# HSV color ranges for chess.com square backgrounds (default green theme)
# Includes normal, highlighted (last move), and selected variants
# Ranges for BOARD DETECTION (find_board_by_colors)
# Covers chess.com green theme + highlighted/selected squares.
BOARD_DETECT_COLOR_RANGES = [
    # Normal light squares (cream/beige)
    (np.array([13, 0, 165]), np.array([47, 100, 255])),
    # Normal dark squares (green/olive)
    (np.array([23, 35, 55]), np.array([82, 235, 205])),
    # Highlighted light (more saturated yellow-green)
    (np.array([20, 80, 170]), np.array([50, 160, 255])),
    # Highlighted dark (brighter green)
    (np.array([25, 60, 100]), np.array([60, 255, 220])),
    # Opponent move highlight — teal/cyan (H≈100, S≈110, V≈210)
    (np.array([85, 50, 150]), np.array([115, 180, 255])),
]

# Tight ranges for PIECE DETECTION (background masking) — avoid eating white pieces
SQUARE_COLOR_RANGES = [
    # Normal light (cream) — tight S upper bound to avoid matching white piece pixels
    (np.array([15, 15, 185]), np.array([45, 80, 255])),
    # Normal dark (green/olive)
    (np.array([25, 40, 60]), np.array([82, 210, 195])),
    # Highlighted light (yellow/green tint)
    (np.array([18, 60, 170]), np.array([55, 255, 255])),
    # Highlighted dark (olive)
    (np.array([22, 60, 110]), np.array([60, 255, 220])),
    # Opponent move highlight — teal/cyan
    (np.array([85, 50, 150]), np.array([115, 180, 255])),
]


def _board_sanity_check(image, x, y, w, h):
    """
    Verify a candidate board region by checking piece colors form a
    plausible chess position. Fast — no template matching.
    Returns True if the region looks like a real chessboard.
    """
    cell_w = w / 8
    cell_h = h / 8

    whites, blacks, empty = 0, 0, 0
    ranks_with_pieces = set()

    for row in range(8):
        for col in range(8):
            x1 = int(x + col * cell_w)
            y1 = int(y + row * cell_h)
            x2 = int(x1 + cell_w)
            y2 = int(y1 + cell_h)
            if y2 > image.shape[0] or x2 > image.shape[1]:
                continue
            cell = image[y1:y2, x1:x2]
            color, ratio = classify_cell_color(cell)
            if color == "w":
                whites += 1
                ranks_with_pieces.add(row)
            elif color == "b":
                blacks += 1
                ranks_with_pieces.add(row)
            else:
                empty += 1

    total = whites + blacks

    if total < 4 or total > 34:
        return False
    if whites < 1 or blacks < 1:
        return False
    if len(ranks_with_pieces) < 3:
        return False
    if empty < 20:
        return False
    if total >= 10 and whites >= 3 and blacks >= 3:
        return True

    return False


def find_board_by_colors(image):
    """
    Find the chessboard on screen by detecting chess.com's
    checkerboard pattern. Resolution-independent.
    Handles normal, highlighted, and selected square colors.
    Returns (x, y, width, height) of the board, or None.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Combine all square color masks (use wide ranges for board finding)
    board_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in BOARD_DETECT_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        board_mask = cv2.bitwise_or(board_mask, mask)

    # Find contours with light morphology to avoid merging with UI
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = 350 * 350  # Minimum board size — excludes preview thumbnails
    scored_candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        ratio = w / h if h > 0 else 0
        # Generate square candidates from this contour
        candidates = []

        if 0.85 < ratio < 1.18:
            # Nearly square — use directly and as square-forced
            candidates.append((x, y, w, h))
            side = min(w, h)
            if abs(w - h) > 4:
                # Try alignments for the square version
                for frac in [0.0, 0.5, 1.0]:
                    if h > w:
                        dy = int((h - side) * frac)
                        candidates.append((x, y + dy, side, side))
                    else:
                        dx = int((w - side) * frac)
                        candidates.append((x + dx, y, side, side))

        # If contour is large and non-square, try sub-regions
        # (handles case where board merged with sidebar)
        if area > min_area * 4:
            side = min(w, h)
            for size_frac in [0.5, 0.6, 0.75]:
                sub_side = int(side * size_frac)
                if sub_side * sub_side < min_area:
                    continue
                for fx in [0.0, 0.5, 1.0]:
                    for fy in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        sx = x + int((w - sub_side) * fx)
                        sy = y + int((h - sub_side) * fy)
                        if sx + sub_side <= image.shape[1] and sy + sub_side <= image.shape[0]:
                            candidates.append((sx, sy, sub_side, sub_side))

        for cx, cy, cw, ch in candidates:
            if cx < 0 or cy < 0 or cx + cw > image.shape[1] or cy + ch > image.shape[0]:
                continue
            score = _validate_checkerboard(image, cx, cy, cw, ch)
            if score >= 0.4:
                scored_candidates.append((score, cx, cy, cw, ch))

    if not scored_candidates:
        return None

    # Sort by checkerboard score, then sanity-check top candidates
    scored_candidates.sort(key=lambda t: t[0], reverse=True)

    for score, cx, cy, cw, ch in scored_candidates[:20]:
        # Fine-tune alignment using checkerboard pattern
        best = (cx, cy, cw, ch)
        best_score = score
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                nx, ny = cx + dx, cy + dy
                if nx >= 0 and ny >= 0 and nx + cw <= image.shape[1] and ny + ch <= image.shape[0]:
                    s = _validate_checkerboard(image, nx, ny, cw, ch)
                    if s > best_score:
                        best_score = s
                        best = (nx, ny, cw, ch)

        bx, by, bw, bh = best
        if _board_sanity_check(image, bx, by, bw, bh):
            return best

    # Fallback: return the top scoring candidate without sanity check
    _, cx, cy, cw, ch = scored_candidates[0]
    return (cx, cy, cw, ch)


def _classify_pixel_square_color(hsv_pixel):
    """Classify an HSV pixel as light square, dark square, or unknown."""
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
    # Light squares: cream/beige (normal or highlighted)
    is_light = ((15 <= h <= 50 and s < 100 and v > 170) or
                (20 <= h <= 50 and s > 60 and v > 170))
    # Dark squares: green (normal or highlighted olive)
    is_dark = ((25 <= h <= 85 and s > 40 and 60 < v < 200) or
               (25 <= h <= 60 and s > 60 and 100 < v < 220))
    if is_light:
        return "light"
    if is_dark:
        return "dark"
    return None


def _validate_checkerboard(image, x, y, w, h):
    """
    Validate that a region contains an actual 8x8 checkerboard pattern.
    Uses two methods:
    1. Cell-based: samples corners, checks alternating light/dark pattern
    2. Transition-based: counts color transitions along scan lines (should be ~7 per line)
    The transition check rejects regions at wrong scales (2x board would have ~14 transitions).
    Returns a confidence score 0-1.
    """
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cell_w = w / 8
    cell_h = h / 8

    # --- Method 1: Cell-based alternating pattern ---
    alternating_count = 0
    total_checks = 0
    offsets = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]

    for row in range(8):
        for col in range(8):
            expected_light = (row + col) % 2 == 0
            votes_light = 0
            votes_dark = 0
            for ox, oy in offsets:
                px = int(col * cell_w + cell_w * ox)
                py = int(row * cell_h + cell_h * oy)
                if py >= hsv.shape[0] or px >= hsv.shape[1]:
                    continue
                c = _classify_pixel_square_color(hsv[py, px])
                if c == "light":
                    votes_light += 1
                elif c == "dark":
                    votes_dark += 1
            if votes_light + votes_dark >= 2:
                total_checks += 1
                is_light = votes_light > votes_dark
                if (expected_light and is_light) or (not expected_light and not is_light):
                    alternating_count += 1

    if total_checks < 16:
        return 0.0
    pattern_score = alternating_count / total_checks

    # --- Method 2: Transition spacing verification ---
    # Scan lines and check that transitions happen at positions consistent
    # with the assumed cell size (w/8). If the board is at wrong scale,
    # transitions will be spaced at a different interval.
    expected_cell = w / 8  # Expected cell width in pixels
    spacing_scores = []
    sample_fracs = [0.15, 0.35, 0.5, 0.65, 0.85]

    for frac in sample_fracs:
        for horizontal in [True, False]:
            if horizontal:
                scan_len = w
                fixed = int(h * frac)
                if fixed >= hsv.shape[0]:
                    continue
            else:
                scan_len = h
                fixed = int(w * frac)
                if fixed >= hsv.shape[1]:
                    continue

            # Find transition positions
            transitions = []
            last_class = None
            step = max(1, scan_len // 100)
            for i in range(0, scan_len, step):
                if horizontal:
                    if i >= hsv.shape[1]:
                        break
                    c = _classify_pixel_square_color(hsv[fixed, i])
                else:
                    if i >= hsv.shape[0]:
                        break
                    c = _classify_pixel_square_color(hsv[i, fixed])
                if c is not None:
                    if last_class is not None and c != last_class:
                        transitions.append(i)
                    last_class = c

            if len(transitions) < 4:
                continue

            # Check spacing between consecutive transitions
            spacings = [transitions[i+1] - transitions[i] for i in range(len(transitions)-1)]
            if not spacings:
                continue
            median_spacing = sorted(spacings)[len(spacings) // 2]

            # Score: how close is the median spacing to the expected cell size?
            spacing_ratio = median_spacing / expected_cell if expected_cell > 0 else 0
            # Perfect = 1.0, double = 0.5 or 2.0 → both wrong
            s = max(0, 1.0 - abs(spacing_ratio - 1.0))
            # Also check count: should be 5-9 transitions (7 ideal, some may be missed)
            count_score = max(0, 1.0 - abs(len(transitions) - 7) / 5)
            spacing_scores.append(s * 0.7 + count_score * 0.3)

    if not spacing_scores:
        return pattern_score * 0.5

    spacing_score = sum(spacing_scores) / len(spacing_scores)

    # Combined: pattern + spacing must both be good
    return pattern_score * 0.5 + spacing_score * 0.5


def get_background_mask(cell_img):
    """
    Create a mask of background (square color) pixels.
    Handles normal, highlighted, selected, and check-highlighted squares.
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    bg_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in SQUARE_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        bg_mask = cv2.bitwise_or(bg_mask, mask)

    # Also catch check highlights (reddish squares)
    red_lower1 = np.array([0, 50, 100])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 50, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2),
    )
    bg_mask = cv2.bitwise_or(bg_mask, red_mask)

    return bg_mask


def classify_cell_color(cell_img):
    """
    Determine if a cell has a piece and whether it's white or black.
    Handles all chess.com square color variants (normal, highlighted, check).
    Uses the inner region of the cell to avoid edge artifacts.
    """
    bg_mask = get_background_mask(cell_img)
    total = cell_img.shape[0] * cell_img.shape[1]
    bg_count = cv2.countNonZero(bg_mask)
    piece_ratio = 1 - (bg_count / total)

    if piece_ratio < config["piece_detect_threshold"]:
        return "", piece_ratio

    # Use inner 60% of cell to avoid background bleed at edges
    h, w = cell_img.shape[:2]
    margin_y = int(h * 0.2)
    margin_x = int(w * 0.2)
    inner = cell_img[margin_y:h-margin_y, margin_x:w-margin_x]

    if inner.size == 0:
        return "", piece_ratio

    # Check brightness of inner region (ignoring background-colored pixels)
    inner_bg = get_background_mask(inner)
    inner_piece = cv2.bitwise_not(inner_bg)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    nonzero = gray[inner_piece > 0]

    if len(nonzero) == 0:
        # No non-background pixels in center — use full cell brightness
        piece_mask = cv2.bitwise_not(bg_mask)
        full_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        nonzero = full_gray[piece_mask > 0]
        if len(nonzero) == 0:
            return "", piece_ratio

    # Use median instead of mean — more robust to dark outlines on white pieces
    median_brightness = np.median(nonzero)
    # Also check: what fraction of piece pixels are "bright" (>150)?
    bright_ratio = np.sum(nonzero > 150) / len(nonzero)

    # White pieces: mostly bright pixels. Black pieces: mostly dark.
    if median_brightness > 140 or bright_ratio > 0.4:
        return "w", piece_ratio
    else:
        return "b", piece_ratio


# ─── Piece Identification ─────────────────────────────────────────────────────

def load_piece_templates():
    """Load all piece template images from script directory."""
    templates = {}
    template_files = {
        "br": "black_rook.png", "bn": "black_knight.png",
        "bb": "black_bishop.png", "bq": "black_queen.png",
        "bk": "black_king.png", "bp": "black_pawn.png",
        "wr": "white_rook.png", "wn": "white_knight.png",
        "wb": "white_bishop.png", "wq": "white_queen.png",
        "wk": "white_king.png", "wp": "white_pawn.png",
    }
    alt_files = {
        "wr_alt": "white_rook_withwhitebackground.png",
        "wn_alt": "white_knight_withgreenbackground.png",
        "wb_alt": "white_bishop_withgreenbackground.png",
        "wq_alt": "white_queen_withgreenbackground.png",
        "wk_alt": "white_king_withwhitebackground.png",
        "wp_alt": "white_pawn_withgreenbackground.png",
    }
    for key, fname in {**template_files, **alt_files}.items():
        path = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                templates[key] = img
    return templates


def match_piece_in_cell(cell_img, templates, piece_color, cell_size):
    """
    Identify piece type using per-cell template matching.
    Templates are auto-scaled to match the detected cell size.
    """
    prefix = "w" if piece_color == "w" else "b"
    candidates = {}
    for key, tmpl in templates.items():
        base = key.replace("_alt", "")
        if base.startswith(prefix):
            ptype = base[1]
            if ptype not in candidates:
                candidates[ptype] = []
            candidates[ptype].append(tmpl)

    best_score = -1
    best_type = "p"
    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    for ptype, tmpls in candidates.items():
        for tmpl in tmpls:
            th, tw = tmpl.shape[:2]
            scale = (cell_size * 0.9) / max(th, tw)
            nw, nh = int(tw * scale), int(th * scale)
            if nw <= 0 or nh <= 0 or nw >= cell_img.shape[1] or nh >= cell_img.shape[0]:
                continue
            tmpl_gray = cv2.cvtColor(cv2.resize(tmpl, (nw, nh)), cv2.COLOR_BGR2GRAY)
            for es in config["template_match_scales"]:
                sw, sh = int(nw * es), int(nh * es)
                if sw <= 0 or sh <= 0 or sw >= cell_img.shape[1] or sh >= cell_img.shape[0]:
                    continue
                scaled = cv2.resize(tmpl_gray, (sw, sh))
                result = cv2.matchTemplate(cell_gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, _ = cv2.minMaxLoc(result)
                if maxv > best_score:
                    best_score = maxv
                    best_type = ptype

    if piece_color == "w":
        pmap = {"r": "R", "n": "N", "b": "B", "q": "Q", "k": "K", "p": "P"}
    else:
        pmap = {"r": "r", "n": "n", "b": "b", "q": "q", "k": "k", "p": "p"}
    return pmap.get(best_type, ""), best_score


# ─── Board Extraction ─────────────────────────────────────────────────────────

def extract_board_from_image(full_image, piece_templates, board_bounds, playing_color,
                             color_only=False):
    """
    Extract board state using color-based detection + per-cell template matching.
    If color_only=True, skips template matching and returns just piece colors
    (e.g., "W" for white piece, "B" for black piece) — used as fallback for
    small cells or when template matching is unreliable.
    Returns: board array (8x8), annotated image, piece count
    """
    bx, by, bw, bh = board_bounds
    cell_w = bw / 8
    cell_h = bh / 8
    cell_size = int((cell_w + cell_h) / 2)

    board = [["" for _ in range(8)] for _ in range(8)]
    piece_count = 0
    annotated = full_image.copy()

    # Draw board grid
    for i in range(9):
        x = int(bx + i * cell_w)
        cv2.line(annotated, (x, by), (x, by + bh), (0, 255, 0), 1)
        y = int(by + i * cell_h)
        cv2.line(annotated, (bx, y), (bx + bw, y), (0, 255, 0), 1)

    for row in range(8):
        for col in range(8):
            x1 = int(bx + col * cell_w)
            y1 = int(by + row * cell_h)
            x2 = int(x1 + cell_w)
            y2 = int(y1 + cell_h)
            cell_img = full_image[y1:y2, x1:x2]

            color, ratio = classify_cell_color(cell_img)
            if not color:
                continue

            if color_only:
                # Just record that there's a piece of this color
                piece = "?" if color == "w" else "?"  # Placeholder
                # Use uppercase/lowercase to track color
                piece = "W" if color == "w" else "B"
            else:
                piece, match_score = match_piece_in_cell(cell_img, piece_templates, color, cell_size)
                # Occupancy gate: an overlay (last-move highlight, game-start "VS" splash,
                # red check-glow) sitting on an EMPTY square can push classify_cell_color
                # over its occupancy threshold and invent a phantom piece. Phantoms assemble
                # into illegal boards (e.g. two kings of one colour) that fail legality and
                # stall the sync loop. A real piece correlates strongly with its template; an
                # overlay does not. So if the best template match is very weak, the square is
                # empty. Threshold sits well below any real piece's score (kept conservative
                # to never drop a real piece); dropped cells are logged so it can be tuned.
                min_score = config.get("occupancy_match_min", 0.30)
                if match_score < min_score:
                    logging.info(f"Occupancy gate: dropped phantom at r{row}c{col} "
                                 f"(match={match_score:.3f} < {min_score}, colour={color})")
                    continue
                elif match_score < 0.45:
                    # Marginal keep — logged so the real score distribution (esp. of any
                    # phantom that squeaks past the gate) is visible for tuning min_score.
                    logging.info(f"Occupancy gate: marginal keep {piece} at r{row}c{col} "
                                 f"(match={match_score:.3f})")

            if playing_color == "b":
                r, c = 7 - row, 7 - col
            else:
                r, c = row, col
            board[r][c] = piece
            piece_count += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, piece, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return board, annotated, piece_count


# ─── Image Comparison ─────────────────────────────────────────────────────────

def are_images_similar(image1_path, image2_path, threshold=0.05):
    """Check if two screenshots are similar enough that the board hasn't changed."""
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        return False
    if image1.shape != image2.shape:
        return False
    diff = cv2.absdiff(image1, image2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    return (non_zero_count / total_pixels) < threshold


# ─── Stockfish Engine ─────────────────────────────────────────────────────────

# ── Official binary setup: Stockfish + geckodriver ────────────────────────────
# Auto-installs the two native binaries the app needs, on demand, when they aren't
# already present. Downloads ONLY from official hosts, SHA256-verified against the
# GitHub release API's per-asset digest, into a local userspace dir (no sudo).
_SETUP_BIN_DIR = os.path.expanduser("~/.config/chessassistant/bin")
_OFFICIAL_HOSTS = ("github.com", "api.github.com", "objects.githubusercontent.com",
                   "codeload.github.com", "pypi.org", "files.pythonhosted.org")
_EXE = ".exe" if platform.system() == "Windows" else ""


def _host_ok(url):
    from urllib.parse import urlparse
    h = (urlparse(url).hostname or "").lower()
    return any(h == d or h.endswith("." + d) for d in _OFFICIAL_HOSTS)


def _confirm_setup(msg):
    try:
        return input(msg + " [Y/n]: ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def _download_verified(url, sha256, dest):
    """Download `url` (official hosts only) to `dest`, verifying SHA256 if given."""
    import urllib.request, hashlib
    if not _host_ok(url):
        raise ValueError(f"refusing non-official host: {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = hashlib.sha256()
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "chessAssistant-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(65536), b""):
            h.update(chunk)
            f.write(chunk)
    if sha256 and h.hexdigest().lower() != sha256.lower():
        os.remove(tmp)
        raise ValueError(f"SHA256 mismatch for {os.path.basename(dest)}")
    os.replace(tmp, dest)
    return dest


def _gh_release(repo, tag):
    """Fetch a GitHub release via the official API. tag e.g. 'tags/sf_18' or 'latest'."""
    import urllib.request, json
    url = f"https://api.github.com/repos/{repo}/releases/{tag}"
    req = urllib.request.Request(url, headers={"User-Agent": "chessAssistant-setup",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _pick_asset(rel, predicate):
    for a in rel.get("assets", []):
        if predicate(a["name"]):
            sha = (a.get("digest") or "").split(":")[-1] or None
            return a["browser_download_url"], sha, a["name"]
    return None, None, None


def _extract_binary(archive, name_contains, dest):
    """Extract archive (.tar/.tar.gz/.zip), find the executable whose name contains
    `name_contains`, copy it to `dest` (+x), and clean up. Returns dest or None."""
    import tarfile, zipfile, stat, tempfile, shutil
    tmpd = tempfile.mkdtemp(prefix="chessassist_setup_")
    try:
        if archive.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive) as t:
                t.extractall(tmpd)
        elif archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmpd)
        else:
            return None
        best = None
        for root, _, files in os.walk(tmpd):
            for f in files:
                lf = f.lower()
                if name_contains not in lf:
                    continue
                if lf.endswith((".nnue", ".txt", ".md", ".zip", ".tar", ".gz", ".ini", ".html")):
                    continue
                p = os.path.join(root, f)
                if best is None or os.path.getsize(p) > os.path.getsize(best):
                    best = p
        if not best:
            return None
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(best, dest)
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return dest
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
        try:
            os.remove(archive)
        except Exception:
            pass


def _works_uci(exe):
    """True if the binary speaks UCI (weeds out a wrong-CPU 'illegal instruction' build)."""
    import subprocess
    try:
        p = subprocess.run([exe], input="uci\nquit\n", capture_output=True,
                           text=True, timeout=10)
        return "uciok" in (p.stdout + p.stderr).lower()
    except Exception:
        return False


def _cpu_stockfish_variant():
    """Best Stockfish build the CPU supports: avx2 > sse41-popcnt > base (Linux via
    /proc/cpuinfo; assume avx2 elsewhere — a run-test falls back if that's wrong)."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                flags = f.read().lower()
            if "avx2" in flags:
                return "avx2"
            if "sse4_1" in flags and "popcnt" in flags:
                return "sse41-popcnt"
            return ""  # base
    except Exception:
        pass
    return "avx2"


def ensure_stockfish():
    """Return a path to a working Stockfish, downloading the official sf_18 build
    (SHA256-verified) into the local bin dir if none is installed. None on decline/fail."""
    dest = os.path.join(_SETUP_BIN_DIR, "stockfish" + _EXE)
    if os.path.exists(dest) and _works_uci(dest):
        return dest
    system = platform.system()
    namer = {
        "Linux":   lambda v: f"stockfish-ubuntu-x86-64{('-' + v) if v else ''}.tar",
        "Windows": lambda v: f"stockfish-windows-x86-64{('-' + v) if v else ''}.zip",
        "Darwin":  lambda v: ("stockfish-macos-m1-apple-silicon.tar"
                              if platform.machine().lower() in ("arm64", "aarch64")
                              else f"stockfish-macos-x86-64{('-' + v) if v else ''}.tar"),
    }.get(system)
    if not namer:
        print("Auto-download of Stockfish isn't supported on this OS."); return None
    print("\033[93mStockfish not found.\033[0m")
    if not _confirm_setup("Download official Stockfish 18 (github.com/official-stockfish, SHA256-verified)?"):
        return None
    try:
        rel = _gh_release("official-stockfish/Stockfish", "tags/sf_18")
    except Exception as e:
        print(f"  couldn't reach the release API: {e}"); return None
    order = []
    for v in (_cpu_stockfish_variant(), "avx2", "sse41-popcnt", ""):
        if v not in order:
            order.append(v)
    for v in order:
        want = namer(v)
        url, sha, name = _pick_asset(rel, lambda n, w=want: n == w)
        if not url:
            continue
        try:
            print(f"  downloading {name} ...")
            arc = _download_verified(url, sha, os.path.join(_SETUP_BIN_DIR, name))
            exe = _extract_binary(arc, "stockfish", dest)
            if exe and _works_uci(exe):
                print(f"\033[92mStockfish ready:\033[0m {exe}")
                return exe
            print(f"  {name} didn't run on this CPU — trying a more compatible build...")
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("Stockfish auto-setup failed."); return None


def ensure_geckodriver():
    """Return a path to geckodriver, downloading the latest official Mozilla build
    (SHA256-verified) into the local bin dir if none is installed. None on decline/fail."""
    dest = os.path.join(_SETUP_BIN_DIR, "geckodriver" + _EXE)
    if os.path.exists(dest):
        return dest
    system, mach = platform.system(), platform.machine().lower()
    if system == "Linux":
        key = "linux64" if mach in ("x86_64", "amd64") else "linux-aarch64"
    elif system == "Windows":
        key = "win64" if mach in ("amd64", "x86_64") else "win32"
    elif system == "Darwin":
        key = "macos-aarch64" if mach in ("arm64", "aarch64") else "macos"
    else:
        return None
    print("\033[93mgeckodriver not found.\033[0m")
    if not _confirm_setup("Download official geckodriver (github.com/mozilla, SHA256-verified)?"):
        return None
    try:
        rel = _gh_release("mozilla/geckodriver", "latest")
    except Exception as e:
        print(f"  couldn't reach the release API: {e}"); return None
    url, sha, name = _pick_asset(
        rel, lambda n: n.endswith(f"-{key}.tar.gz") or n.endswith(f"-{key}.zip"))
    if not url:
        print("  no matching geckodriver asset found."); return None
    try:
        print(f"  downloading {name} ...")
        arc = _download_verified(url, sha, os.path.join(_SETUP_BIN_DIR, name))
        exe = _extract_binary(arc, "geckodriver", dest)
        if exe:
            print(f"\033[92mgeckodriver ready:\033[0m {exe}")
            return exe
    except Exception as e:
        print(f"  geckodriver setup failed: {e}")
    return None


def _firefox_present():
    if platform.system() == "Windows":
        cands = [os.path.join(os.environ.get("PROGRAMFILES", ""), "Mozilla Firefox", "firefox.exe"),
                 os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Mozilla Firefox", "firefox.exe")]
    elif platform.system() == "Darwin":
        cands = ["/Applications/Firefox.app/Contents/MacOS/firefox"]
    else:
        cands = ["/snap/firefox/current/usr/lib/firefox/firefox", "/usr/bin/firefox",
                 "/usr/lib/firefox/firefox"]
    import shutil
    return any(os.path.exists(c) for c in cands) or bool(shutil.which("firefox"))


def run_setup():
    """`--setup`: check/install everything up front and report."""
    print("\033[1m── Setup check ──\033[0m")
    print("Python packages: OK (the app is running, so imports succeeded)")
    sf = ensure_stockfish()
    print(f"Stockfish:   {sf or 'NOT installed'}")
    gd = ensure_geckodriver()
    print(f"geckodriver: {gd or 'NOT installed'}")
    if _firefox_present():
        print("Firefox:     found")
    else:
        print("Firefox:     \033[93mnot found\033[0m — install it from https://www.mozilla.org/firefox/ "
              "(a browser is too heavy to auto-install).")
    print("\033[1m── done ──\033[0m")


_STOCKFISH_ANNOUNCED = False  # print "Found Stockfish" once, not on every re-init


def initialize_stockfish():
    """Find and initialize the Stockfish chess engine."""

    def find_stockfish_exe():
        import shutil  # used in BOTH the POSIX and Windows branches below
        # Check common Linux/Mac paths first (most likely for this user)
        if platform.system() != "Windows":
            for path in ["/usr/games/stockfish", "/usr/local/bin/stockfish",
                         "/usr/bin/stockfish", "stockfish"]:
                if os.path.exists(path):
                    return path
            # Try which
            found = shutil.which("stockfish")
            if found:
                return found
        else:
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase
                      if os.path.exists(f"{d}:\\")]
            # Windows: check known paths first
            quick_paths = [
                os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), "Stockfish", "stockfish.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"), "Stockfish", "stockfish.exe"),
                "stockfish.exe",
            ]
            quick_paths += [os.path.join(d, "Downloads", "stockfish.exe") for d in drives]
            for path in quick_paths:
                if os.path.exists(path):
                    return path
            # Search the Downloads folder on EVERY drive (+ Program Files) for any
            # stockfish*.exe — handles Stockfish extracted to D:\Downloads etc.
            search_dirs = [os.path.join(d, "Downloads") for d in drives]
            search_dirs += [
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            ]
            for search_dir in search_dirs:
                if not os.path.isdir(search_dir):
                    continue
                try:
                    for root, dirs, files in os.walk(search_dir):
                        for f in files:
                            if f.lower().startswith("stockfish") and f.lower().endswith(".exe"):
                                return os.path.join(root, f)
                        # Don't recurse too deep
                        if root.count(os.sep) - search_dir.count(os.sep) > 4:
                            dirs.clear()
                except (PermissionError, OSError):
                    continue
            # Try PATH
            found = shutil.which("stockfish")
            if found:
                return found
            # Last resort: deep-scan every drive (skip system/huge dirs)
            print("Searching all drives for Stockfish (this can take a moment)...")
            skip = {"windows", "$recycle.bin", "system volume information",
                    "appdata", "node_modules", ".git", "winsxs"}
            for d in drives:
                try:
                    for root, dirs, files in os.walk(d):
                        dirs[:] = [x for x in dirs if x.lower() not in skip]
                        for f in files:
                            if f.lower().startswith("stockfish") and f.lower().endswith(".exe"):
                                return os.path.join(root, f)
                        if root.count(os.sep) - d.count(os.sep) > 5:
                            dirs.clear()
                except (PermissionError, OSError):
                    continue
        return None

    stockfish_path = find_stockfish_exe() or ensure_stockfish()
    if stockfish_path:
        global _STOCKFISH_ANNOUNCED
        if not _STOCKFISH_ANNOUNCED:
            print(f"Found Stockfish at: {stockfish_path}")
            _STOCKFISH_ANNOUNCED = True
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            engine.configure({
                "Threads": config["threads"],
                "Hash": config["hash_size"],
            })
            return engine
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")

    print("Stockfish not found automatically.")
    user_path = input("Path to Stockfish (or 'skip'): ").strip()
    if user_path.lower() == "skip":
        print("Warning: Continuing without Stockfish.")
        return None

    if os.path.exists(user_path):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(user_path)
            engine.configure({
                "Threads": config["threads"],
                "Hash": config["hash_size"],
            })
            return engine
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Not found: {user_path}")
    print("Continuing without Stockfish.")
    return None


# ─── Human-like Timing ────────────────────────────────────────────────────────

def calculate_human_think_time(board, move, best_score, sel_score, move_number,
                                piece_count, clock_remaining=None):
    """
    Calculate realistic human think time scaled to the clock.
    Budget: use ~60-70% of remaining time over ~20 expected remaining moves.
    Then vary based on position complexity and move type.
    """
    # Budget-based max
    if clock_remaining is not None and clock_remaining > 0:
        estimated_moves_left = max(10, 40 - move_number)
        budget = min((clock_remaining * 0.6) / estimated_moves_left, 12.0)
    else:
        budget = 8.0

    # Bimodal human timing: humans either play fast (intuition) or slow (calculation)
    # Not a uniform distribution — it's two peaks
    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    legal_count = len(list(board.legal_moves))

    # Decide: fast move or slow think?
    fast_chance = 0.35  # Base 35% chance of fast/intuitive move
    if is_capture and legal_count <= 5:
        fast_chance = 0.85  # Forced recapture: almost always fast
    elif is_capture:
        fast_chance = 0.60  # Captures: often quick
    elif is_check:
        fast_chance = 0.70  # Giving check: usually pre-planned
    elif legal_count <= 3:
        fast_chance = 0.80  # Few options: quick decision
    elif legal_count >= 30:
        fast_chance = 0.15  # Many options: need to think
    elif move_number <= 8:
        fast_chance = 0.65  # Opening: theory, fast

    if random.random() < fast_chance:
        # Fast move: 0.3-2.0s (intuition/premove/theory)
        base = random.uniform(0.3, 2.0)
    else:
        # Slow think: 2.0-8.0s (calculation)
        base = random.uniform(2.0, min(8.0, budget * 0.8))

    # Occasional very long think (3% — staring at a critical position)
    if random.random() < 0.03 and (clock_remaining is None or clock_remaining > 120):
        base = random.uniform(6.0, min(15.0, budget * 0.9))

    # Occasional instant move (12% — premove or obvious recapture)
    if random.random() < 0.12:
        base = random.uniform(0.2, 0.8)

    # Clock pressure — hard overrides
    if clock_remaining is not None:
        if clock_remaining < 10:
            base = min(base, random.uniform(0.1, 0.4))
        elif clock_remaining < 30:
            base = min(base, random.uniform(0.2, 1.0))
        elif clock_remaining < 60:
            base = min(base, random.uniform(0.4, 2.0))
        elif clock_remaining < 120:
            base = min(base, random.uniform(0.6, 3.0))

    return max(0.2, min(base, budget))


def get_clock_remaining(selenium_ctrl):
    """Try to read the player's clock from chess.com via Selenium."""
    if not selenium_ctrl or not selenium_ctrl.is_alive():
        return None
    try:
        from selenium.webdriver.common.by import By
        # Chess.com clock selectors (bottom clock = our clock)
        for sel in ['.clock-bottom', '.clock-player-turn',
                    '[data-cy="clock-bottom"]', '.clock-component.clock-bottom']:
            elems = selenium_ctrl.driver.find_elements(By.CSS_SELECTOR, sel)
            for elem in elems:
                if elem.is_displayed():
                    text = elem.text.strip()
                    # Parse "5:00" or "1:23.4" or "0:45"
                    parts = text.replace('.', ':').split(':')
                    if len(parts) >= 2:
                        try:
                            mins = int(parts[0])
                            secs = int(parts[1])
                            return mins * 60 + secs
                        except ValueError:
                            pass
    except Exception:
        pass
    return None


# ─── Opening Book ─────────────────────────────────────────────────────────────

# Common openings — plays these from memory (no engine) for the first few moves.
# This is undetectable because it's exactly what humans do.
OPENING_BOOK = {
    # As White
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR": [
        "e2e4", "d2d4", "c2c4", "g1f3",
    ],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR": [  # After 1.e4
        "g1f3", "d2d4", "f1c4", "b1c3",
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR": [  # After 1.d4
        "c2c4", "g1f3", "c1f4", "b1c3",
    ],
    # As Black vs 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR": [
        "e7e5", "c7c5", "e7e6", "c7c6", "d7d5",
    ],
    # As Black vs 1.d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR": [
        "d7d5", "g8f6", "e7e6", "f7f5",
    ],
    # As Black vs 1.c4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR": [
        "e7e5", "g8f6", "c7c5", "e7e6",
    ],
    # As Black vs 1.Nf3
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R": [
        "d7d5", "g8f6", "c7c5",
    ],
}


def get_book_move(board):
    """Try to get a move from the opening book. Returns move or None."""
    # Use book for up to 12 moves (24 half-moves)
    if len(board.move_stack) > 24:
        return None
    # Gradually reduce book usage: 100% at move 1, ~50% at move 8, ~20% at move 12
    book_chance = max(0.15, 1.0 - len(board.move_stack) * 0.035)
    if random.random() > book_chance:
        return None

    # Get position key (just the piece placement, ignore castling/en passant)
    fen_parts = board.fen().split()
    position = fen_parts[0]

    candidates = OPENING_BOOK.get(position, [])
    if not candidates:
        return None

    # Filter to legal moves
    legal = []
    for uci in candidates:
        try:
            move = chess.Move.from_uci(uci)
            if board.is_legal(move):
                legal.append(move)
        except Exception:
            pass

    if legal:
        return random.choice(legal)
    return None


# ─── Move Analysis ────────────────────────────────────────────────────────────

def get_best_move(engine, board, piece_count, use_randomizer, skill_level,
                  opponent_rating=None, opp_avg_cpl=None, retries=3):
    """
    Analyze position and select a move with anti-detection mechanics.
    Uses the python-chess Board directly for accurate position info.
    """
    # Nothing to analyse once the game is over (checkmate/stalemate): engine.analyse
    # errors on a terminal board, which used to kick off a noisy engine re-init loop
    # (the watch-mode "Found Stockfish" spam). Bail cleanly instead.
    if board.is_game_over():
        return None, None, None
    for attempt in range(retries):
        try:
            # Adaptive depth with occasional shallow analysis (creates natural errors)
            # 12% chance of "lazy" analysis — humans don't always calculate deeply
            if random.random() < 0.12:
                base_depth = random.randint(4, 8)
                min_depth = 4
            elif piece_count <= 16:
                base_depth = random.randint(12, 18)
                min_depth = 12
            elif piece_count <= 25:
                base_depth = random.randint(10, 16)
                min_depth = 10
            else:
                base_depth = random.randint(10, 14)
                min_depth = 10

            entropy = 0.3

            # Quick analysis for critical position detection + entropy
            try:
                quick = engine.analyse(board, chess.engine.Limit(depth=10), multipv=3)
                scores = [m["score"].relative.score() for m in quick
                          if m["score"].relative.score() is not None]
                if scores:
                    if max(abs(s) for s in scores) > 150:
                        base_depth = max(base_depth, 20)
                    entropy = (max(scores) - min(scores)) / 100
            except Exception as e:
                logging.error(f"Quick analysis failed: {e}")

            depth = min(max(base_depth + (1 if entropy > 0.3 else 0), min_depth),
                        config["max_depth"])

            logging.info(f"Analysis depth={depth}, entropy={entropy:.2f}")

            try:
                analysis = engine.analyse(
                    board, chess.engine.Limit(depth=depth),
                    multipv=5, options={"Skill Level": skill_level},
                )
            except Exception:
                analysis = engine.analyse(
                    board, chess.engine.Limit(time=2.0 if piece_count <= 16 else 1.0),
                    multipv=5, options={"Skill Level": skill_level},
                )

            if not analysis:
                raise Exception("No analysis returned")

            best_move = analysis[0]["pv"][0]
            best_score = analysis[0]["score"].relative.score() or 0

            if not best_move or not board.is_legal(best_move):
                raise Exception("Invalid move generated")

            # Select from top moves — window for human-like play
            advantage = best_score / 100
            move_num = len(board.move_stack) // 2 + 1

            # When we're clearly winning, CONVERT: play accurately instead of loosening
            # up. A human up material plays their good moves — that isn't suspicious,
            # whereas throwing a won game (especially on time) is the most bot-like
            # outcome there is. Camouflage (wide windows, blunders, CPL-matching) is
            # reserved for equal/worse positions, where "too perfect" would stand out.
            converting = advantage >= 2.5

            # Base window: 100-200cp (humans regularly play 1-2 pawn suboptimal)
            if converting:
                max_diff = 45   # tight — near-best variety only, don't hand back the win
            elif advantage < -3.0:
                max_diff = 120 + abs(advantage) * 10  # losing → fight, take chances
            elif abs(advantage) > 1.5:
                max_diff = 100 + abs(advantage) * 8
            else:
                max_diff = 80 + entropy * 25

            # Opponent-adaptive: match ~70% of their CPL — but never while converting a win
            if not converting and opp_avg_cpl is not None and opp_avg_cpl > 30:
                target_cpl = opp_avg_cpl * 0.7
                max_diff = max(max_diff, target_cpl * 2.0)
                max_diff = min(max_diff, 300)

            # Fatigue: window widens as game goes on — but not when converting
            if not converting and move_num > 20:
                max_diff *= 1.0 + (move_num - 20) * 0.02

            valid_moves = []
            for m in analysis:
                ms = m["score"].relative.score() or 0
                diff = abs(ms - best_score)
                if diff <= max_diff:
                    valid_moves.append((m, diff))

            if not valid_moves:
                return best_move, best_score, best_score

            # Blunder injection — realistic rates, but NEVER sabotage a winning position
            blunder_chance = 0.06  # Base 6% (humans blunder often)
            if move_num > 20:
                blunder_chance += (move_num - 20) * 0.008  # +0.8% per move after 20
            if advantage < -2.0:
                blunder_chance += 0.04  # Desperation when losing
            if converting:
                blunder_chance = 0.0   # up big → just convert, don't hand it back

            if use_randomizer and blunder_chance > 0 and random.random() < min(blunder_chance, 0.25):
                # Pick a "human-like" blunder: prefer captures and checks
                # (humans blunder into tactics, not random squares)
                all_legal = list(board.legal_moves)
                non_best = [m for m in all_legal if m != best_move]
                if non_best:
                    # Weight toward captures/checks (humans blunder tactically)
                    blunder_candidates = []
                    for m in non_best:
                        w = 1.0
                        if board.is_capture(m):
                            w = 3.0  # Captures look tempting
                        if board.gives_check(m):
                            w = 2.5  # Checks feel aggressive
                        blunder_candidates.append((m, w))
                    blunder_weights = [w for _, w in blunder_candidates]
                    blunder = random.choices([m for m, _ in blunder_candidates],
                                             weights=blunder_weights, k=1)[0]
                    logging.info(f"Blunder injected: {blunder}")
                    return blunder, best_score, best_score - 100

            # Exponential weighting (not linear) — makes best move less dominant
            # Linear: best=1.0, 50cp=0.5, 100cp=0 → best wins 2:1
            # Exponential: best=1.0, 50cp=0.37, 100cp=0.13 → more spread
            import math
            decay = 0.015  # Lower = more spread, higher = more concentrated
            weights = [math.exp(-decay * d) * random.uniform(0.7, 1.3)
                       for _, d in valid_moves]

            # 20% chance of random pick in equal positions (was 10%)
            if random.random() < 0.20 and abs(advantage) < 1.5:
                sel = random.choice(valid_moves)[0]
            else:
                sel = random.choices([m for m, _ in valid_moves], weights=weights, k=1)[0]

            sel_score = sel["score"].relative.score() or 0
            sel_move = sel["pv"][0]

            rank = analysis.index(sel) + 1
            logging.info(f"Selected: {sel_move} (rank {rank}, score {sel_score})")

            return sel_move, best_score, sel_score

        except Exception as e:
            logging.error(f"Analysis attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                try:
                    try:
                        engine.quit()   # don't leak the dead engine process
                    except Exception:
                        pass
                    engine = initialize_stockfish()
                except Exception:
                    pass
            time.sleep(0.5)

    # Emergency: pick first legal move
    legal = list(board.legal_moves)
    if legal:
        logging.warning(f"Emergency move: {legal[0]}")
        return legal[0], 0, 0
    return None, None, None


# ─── Move Display ─────────────────────────────────────────────────────────────

def format_move(move, board):
    """Format a move for display with piece name and squares."""
    piece = board.piece_at(move.from_square)
    piece_name = ""
    if piece:
        names = {"P": "Pawn", "N": "Knight", "B": "Bishop",
                 "R": "Rook", "Q": "Queen", "K": "King"}
        piece_name = names.get(piece.symbol().upper(), "")

    src = chess.SQUARE_NAMES[move.from_square]
    dst = chess.SQUARE_NAMES[move.to_square]
    promo = ""
    if move.promotion:
        promo_names = {chess.QUEEN: "Q", chess.ROOK: "R",
                       chess.BISHOP: "B", chess.KNIGHT: "N"}
        promo = f" (promote to {promo_names.get(move.promotion, '?')})"

    return f"\033[1m\033[92m▶ {piece_name} {src} → {dst}{promo}\033[0m"


def format_eval(score):
    """Format an engine evaluation score."""
    if score is None:
        return "?"
    pawns = score / 100
    if pawns > 0:
        return f"\033[92m+{pawns:.1f}\033[0m"
    elif pawns < 0:
        return f"\033[91m{pawns:.1f}\033[0m"
    return "0.0"


def print_status(game_state, move=None, score=None):
    """Print a compact status line."""
    stage = game_state.game_stage
    pieces = game_state.piece_count
    move_num = len(game_state.move_history) // 2 + 1
    turn = "White" if game_state.board.turn == chess.WHITE else "Black"

    parts = [f"Move {move_num}", f"{turn} to play", f"{stage} ({pieces} pcs)"]
    if score is not None:
        parts.append(f"eval: {format_eval(score)}")

    print(f"\033[90m{'  |  '.join(parts)}\033[0m")
    if move:
        print(format_move(move, game_state.board))


# ─── Skill Adjustment ─────────────────────────────────────────────────────────

def adjust_skill_level(engine, current_skill, piece_count, score_diff, game=None,
                       our_eval=None):
    """Dynamically adjust skill level to play slightly above opponent's level.
    Uses opponent's average CPL to estimate their ELO and match it. Never drops so
    low that it hangs pieces (min 11), and holds skill HIGH while we're clearly
    winning so a won game gets converted instead of thrown back."""
    min_skill = 11   # was 5 — sub-10 hangs pieces and reads as broken, not human

    # Estimate target skill from opponent's play quality
    target_skill = current_skill
    if game and hasattr(game, '_opp_cpls') and len(game._opp_cpls) >= 3:
        avg_cpl = sum(game._opp_cpls) / len(game._opp_cpls)
        # Map opponent CPL to our skill level (play slightly above them, but floored)
        # CPL <20 = engine-level → skill 20;  higher CPL = weaker → lower skill, but
        # never below ~12 (a floored strong-club level still won't look engine-like
        # thanks to the move-window camouflage, and it stops throwing pieces).
        if avg_cpl < 20:
            target_skill = 20
        elif avg_cpl < 40:
            target_skill = random.randint(17, 20)
        elif avg_cpl < 60:
            target_skill = random.randint(15, 18)
        elif avg_cpl < 90:
            target_skill = random.randint(13, 16)
        elif avg_cpl < 130:
            target_skill = random.randint(12, 15)
        else:
            target_skill = random.randint(min_skill, 14)

    # Clearly winning (>= ~2.5 pawns) → hold skill high and convert, rather than
    # dropping to "match" a weak opponent while we're up material (that's exactly how
    # the won games in testing got thrown).
    if our_eval is not None and our_eval >= 250:
        target_skill = max(target_skill, 17)

    # Drift toward target (±2 per move)
    diff_to_target = target_skill - current_skill
    if abs(diff_to_target) > 3:
        step = random.randint(2, 3) * (1 if diff_to_target > 0 else -1)
    elif abs(diff_to_target) > 0:
        step = random.randint(1, 2) * (1 if diff_to_target > 0 else -1)
    else:
        step = 0
    new_skill = current_skill + step

    # Random variation (human-like inconsistency, 20% chance) — gentler downside
    if random.random() < 0.20:
        new_skill += random.choice([-2, -1, 1, 2, 3])

    new_skill = max(min_skill, min(20, new_skill))

    if new_skill != current_skill:
        engine.configure({"Skill Level": new_skill})
        logging.info(f"Skill level: {current_skill} → {new_skill} (opp target: {target_skill})")

    return new_skill


# ─── Win/Loss Detection ──────────────────────────────────────────────────────

def check_win(image):
    """Check for win indicators (trophy, crown) in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for name in ["trophy.png", "crown.png"]:
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if tmpl is not None:
                tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                for scale in [0.75, 1.0, 1.25]:
                    th, tw = tmpl_gray.shape
                    sw, sh = int(tw * scale), int(th * scale)
                    if sw < 10 or sh < 10 or sw > gray.shape[1] or sh > gray.shape[0]:
                        continue
                    scaled = cv2.resize(tmpl_gray, (sw, sh))
                    res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                    if np.max(res) >= 0.82:
                        return True
    return False


# ─── Auto-Move (click pieces on screen) ──────────────────────────────────────

def get_square_screen_coords(square, board_bounds, playing_color):
    """
    Get the screen pixel coordinates for the center of a chess square.
    square: chess.Square (0-63)
    """
    bx, by, bw, bh = board_bounds
    cell_w = bw / 8
    cell_h = bh / 8

    file = chess.square_file(square)  # 0=a, 7=h
    rank = chess.square_rank(square)  # 0=1, 7=8

    if playing_color == "w":
        col = file
        row = 7 - rank
    else:
        col = 7 - file
        row = rank

    x = int(bx + col * cell_w + cell_w / 2)
    y = int(by + row * cell_h + cell_h / 2)
    return x, y


class SeleniumController:
    """Controls chess.com via Selenium — clicks squares directly in the browser DOM."""

    def __init__(self):
        self.driver = None
        self.board_elem = None
        self.ready = False  # True once user is logged in and game page is loaded

    def launch(self, game_mode=None, playing_color="w", auto_start=True):
        """Launch a Selenium-controlled Firefox and open chess.com.
        auto_start=False navigates only (used by the read-only probe)."""
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service

        opts = Options()
        # Always launch our OWN Firefox instance, even if the user's Firefox is
        # already open. Without this, Firefox hands the launch off to the running
        # copy and the automated process exits immediately ("closed with status 0").
        opts.add_argument("-no-remote")
        opts.add_argument("-new-instance")
        if platform.system() == "Windows":
            import string
            _drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            browser_paths = [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Mozilla Firefox", "firefox.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Mozilla Firefox", "firefox.exe"),
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "Mozilla Firefox", "firefox.exe"),
            ]
            gecko_candidates = ["geckodriver.exe"]
            gecko_candidates += [os.path.join(d, "Downloads", "geckodriver.exe") for d in _drives]
        else:
            browser_paths = [
                '/snap/firefox/current/usr/lib/firefox/firefox',
                '/usr/bin/firefox', '/usr/lib/firefox/firefox',
            ]
            gecko_candidates = [
                '/snap/bin/geckodriver', '/usr/bin/geckodriver',
                '/usr/local/bin/geckodriver',
            ]

        for binary in browser_paths:
            if os.path.exists(binary):
                opts.binary_location = binary
                break

        import shutil
        gecko = next((p for p in gecko_candidates if os.path.exists(p)), None)
        if gecko is None:
            gecko = shutil.which("geckodriver")
        if gecko is None:
            gecko = ensure_geckodriver()   # offer the official download if still missing
        if gecko:
            service = Service(executable_path=gecko)
        else:
            service = Service()

        # Anti-fingerprinting: comprehensive stealth
        opts.set_preference("dom.webdriver.enabled", False)
        opts.set_preference("useAutomationExtension", False)
        opts.set_preference("dom.disable_window_move_resize", False)
        # Disable WebRTC leak (reveals real IP behind VPN)
        opts.set_preference("media.peerconnection.enabled", False)
        # Normal browser behavior
        opts.set_preference("dom.popup_maximum", 10)
        opts.set_preference("privacy.trackingprotection.enabled", True)

        # Persistent profile: reuse ONE profile across runs so a by-hand login
        # (including solving the Cloudflare Turnstile yourself) survives, instead of
        # a fresh throwaway each launch. Uses opts.profile=FirefoxProfile(dir) to
        # dodge the geckodriver 0.36 "-profile" preferences bug (Selenium #14652).
        try:
            from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
            self.profile_dir = os.path.expanduser("~/.config/chessassistant/ff_profile")
            os.makedirs(self.profile_dir, exist_ok=True)
            _profile = FirefoxProfile(self.profile_dir)
            _profile.set_preference("browser.startup.page", 3)                # restore last session
            _profile.set_preference("browser.sessionstore.resume_from_crash", True)
            _profile.set_preference("browser.sessionstore.max_resumed_crashes", -1)
            opts.profile = _profile
        except Exception as e:
            logging.error(f"Persistent profile setup failed, using throwaway: {e}")
            self.profile_dir = None

        self.driver = webdriver.Firefox(options=opts, service=service)

        # Position browser on the left half of the screen
        self._tile_left()

        print(f"\033[92mBrowser opened:\033[0m chess.com")

        if game_mode == "player":
            # Login flow first — no popup dismissal until logged in
            self._wait_for_login_and_play()
            if auto_start:
                self.setup_and_start_game()   # picks the chosen time + Start Game
            self.ready = True
        elif game_mode == "bot":
            self.driver.get('https://www.chess.com/play/computer')
            time.sleep(4)
            self.dismiss_popups()
            self._start_bot_game(playing_color)
            self.ready = True
        elif game_mode == "watch":
            # Watch mode: just open chess.com, user sets up the game
            self.driver.get('https://www.chess.com')
            time.sleep(3)
            self.dismiss_popups()
            print("\033[92mWatch mode:\033[0m waiting for a game to appear on screen...")
            self.ready = True
        else:
            self.driver.get('https://www.chess.com')
            time.sleep(4)
            self.dismiss_popups()
            self.ready = True

        return True

    def _start_bot_game(self, playing_color):
        """Auto-start a bot game on chess.com."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            # Click "Play" or "Start" button if present
            for selector in ['button[data-cy="play-button"]',
                             'a[href*="play/computer"]',
                             'button.cc-button-primary',
                             'button.ui_v5-button-primary']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elems:
                    if elem.is_displayed():
                        elem.click()
                        time.sleep(2)
                        self.dismiss_popups()
                        break

            # Choose color if there's a color selector
            if playing_color == "b":
                for selector in ['[data-cy="black-button"]', 'button[title="Black"]',
                                 '.selection-menu-button-black']:
                    elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elems:
                        if elem.is_displayed():
                            elem.click()
                            time.sleep(1)
                            break

            # Click Play/Start
            time.sleep(1)
            self.dismiss_popups()
            for selector in ['button[data-cy="play-button"]',
                             'button.cc-button-primary',
                             'button.ui_v5-button-primary',
                             'button.ui_v5-button-large']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elems:
                    try:
                        text = elem.text.lower()
                        if elem.is_displayed() and ('play' in text or 'start' in text or 'choose' in text):
                            elem.click()
                            time.sleep(2)
                            self.dismiss_popups()
                            print("\033[92mGame started!\033[0m")
                            return
                    except Exception:
                        pass

            print("Set up your game in the browser, the assistant will detect it automatically.")
        except Exception as e:
            logging.error(f"Auto-start bot game failed: {e}")
            print("Set up your game in the browser manually.")

    def _import_cookies_from_firefox(self):
        """Import login cookies from the user's real Firefox profile."""
        import sqlite3, glob

        profile_dirs = [
            os.path.expanduser("~/.mozilla/firefox"),                      # Linux
            os.path.expanduser("~/snap/firefox/common/.mozilla/firefox"),  # Linux (snap)
            os.path.join(os.environ.get("APPDATA", ""), "Mozilla", "Firefox", "Profiles"),  # Windows
            os.path.expanduser("~/Library/Application Support/Firefox/Profiles"),  # macOS
        ]

        for base in profile_dirs:
            for profile in glob.glob(os.path.join(base, "*.default*")):
                cookie_db = os.path.join(profile, "cookies.sqlite")
                if not os.path.exists(cookie_db):
                    continue
                try:
                    # Copy the DB (Firefox locks it while running)
                    import shutil
                    tmp_db = os.path.join(tempfile.gettempdir(), "chess_cookies.sqlite")
                    shutil.copy2(cookie_db, tmp_db)

                    conn = sqlite3.connect(tmp_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name, value, host, path, isSecure FROM moz_cookies WHERE host LIKE '%chess.com%'")
                    cookies = cursor.fetchall()
                    conn.close()
                    os.remove(tmp_db)

                    if cookies:
                        # Navigate to chess.com first (cookies need matching domain)
                        self.driver.get('https://www.chess.com')
                        time.sleep(2)
                        imported = 0
                        for name, value, host, path, secure in cookies:
                            try:
                                self.driver.add_cookie({
                                    'name': name,
                                    'value': value,
                                    'domain': host,
                                    'path': path,
                                    'secure': bool(secure),
                                })
                                imported += 1
                            except Exception:
                                pass
                        if imported > 0:
                            logging.info(f"Imported {imported} cookies from Firefox profile")
                            return True
                except Exception as e:
                    logging.error(f"Cookie import failed: {e}")
        return False

    def _cookie_store(self):
        return os.path.expanduser("~/.config/chessassistant/chesscom_session.json")

    def _save_cookies(self):
        """Save chess.com cookies from the live session so a by-hand login
        (incl. the one-time Turnstile solve) survives the next launch and we
        can skip the login page entirely."""
        try:
            cookies = self.driver.get_cookies()
        except Exception as e:
            # Session already gone (e.g. browser crashed) — nothing to save, skip quietly.
            if _is_session_dead_error(e):
                logging.info("_save_cookies: session already closed; skipping")
            else:
                logging.error(f"_save_cookies: get_cookies failed: {e}")
            return
        cookies = [c for c in cookies if 'chess.com' in (c.get('domain') or '')]
        if not cookies:
            return
        try:
            path = self._cookie_store()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(cookies, f)
            logging.info(f"Saved {len(cookies)} chess.com cookies for next launch")
        except Exception as e:
            logging.error(f"_save_cookies: write failed: {e}")

    def _load_cookies(self):
        """Restore saved chess.com cookies (from a previous by-hand login) so we
        land already logged in and never hit the login page / Turnstile."""
        path = self._cookie_store()
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                cookies = json.load(f)
        except Exception as e:
            logging.error(f"_load_cookies: read failed: {e}")
            return False
        if not cookies:
            return False
        try:
            self.driver.get('https://www.chess.com')   # must be on-domain to add cookies
            time.sleep(2)
        except Exception:
            pass
        restored = 0
        for c in cookies:
            c.pop('sameSite', None)   # Firefox rejects some sameSite values
            if c.get('expiry') is not None:
                try:
                    c['expiry'] = int(c['expiry'])
                except Exception:
                    c.pop('expiry', None)
            try:
                self.driver.add_cookie(c)
                restored += 1
            except Exception:
                pass
        logging.info(f"Restored {restored}/{len(cookies)} saved cookies")
        return restored > 0

    def _wait_for_login_and_play(self):
        """Use the persistent profile: if it's already logged in from a previous
        by-hand login, go straight to play. Otherwise wait for you to log in by
        hand (you solve the 'Verify you are human' check yourself — that's not
        spoofing). The session is saved to the profile on exit, so it's a one-time
        login. NOTE: importing cookies from your normal Firefox no longer works
        (chess.com's session isn't written to disk since Firefox 152), so we rely
        on this profile instead."""
        # Restore a saved session (from a previous by-hand login) so we skip
        # the login page + Turnstile entirely.
        self._load_cookies()
        # Already logged in from a previous session?
        self.driver.get('https://www.chess.com/home')
        time.sleep(3)
        if 'login' not in self.driver.current_url.lower():
            print("\033[92mLogged in (saved session)!\033[0m")
            self._save_cookies()   # refresh the saved cookies
            self.driver.get('https://www.chess.com/play/online')
            time.sleep(3)
            self.dismiss_popups()
            return

        # Not logged in yet — wait for a manual, human login
        self.driver.get('https://www.chess.com/login')
        time.sleep(2)
        print("\033[93mLog in — or sign up — by hand in the browser window.\033[0m")
        print("\033[90m  Solve the 'Verify you are human' box yourself (you're human, so that's fine).\033[0m")
        print("\033[90m  It won't start a game until you're signed in, and it saves the login for next time.\033[0m")
        while True:
            time.sleep(2)
            try:
                if 'login' not in self.driver.current_url.split('?')[0]:
                    print("\033[92mSigned in!\033[0m Saving session, navigating to play...")
                    self._save_cookies()   # persist this login for next launch
                    self.driver.get('https://www.chess.com/play/online')
                    time.sleep(3)
                    self.dismiss_popups()
                    return
            except Exception:
                pass

    def dismiss_popups(self):
        """Auto-dismiss chess.com popups, modals, cookie banners, etc.
        Skips login/signup forms so user can authenticate."""
        from selenium.webdriver.common.by import By

        # Check if a login/signup form is visible — don't dismiss it
        login_selectors = ['form[action*="login"]', '#login', '.login-modal',
                           'input[name="username"]', 'input[name="password"]',
                           '[data-cy="login-form"]', '.auth-modal']
        for sel in login_selectors:
            try:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if any(e.is_displayed() for e in elems):
                    return  # Login form visible — don't dismiss anything
            except Exception:
                pass

        popup_selectors = [
            # "Play the bots" modal start button
            'button.cc-button-primary',
            # Generic modal close buttons (but NOT on login pages)
            '.modal-close-icon', '.ui_outside-close-icon',
            '[data-cy="modal-close"]',
            # Close X buttons on overlays
            '[aria-label="Close"]', '[aria-label="close"]',
            # Cookie consent
            '#onetrust-accept-btn-handler',
            '[id*="cookie"] button', '[class*="cookie"] button',
            # "Got it" / "OK" / "No thanks" buttons in modals
            '.modal-content button.primary',
            # Chess.com specific overlays
            '.modal-game-over-button', '.modal-first-time-button',
            # Premium/upsell popups
            '[class*="promo"] [aria-label="Close"]',
            '[class*="upgrade"] [aria-label="Close"]',
            '[class*="popup"] [aria-label="Close"]',
        ]
        dismissed = 0
        for selector in popup_selectors:
            try:
                elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elems:
                    if elem.is_displayed():
                        # Skip if it's inside a login/auth context
                        try:
                            parent_html = elem.find_element(By.XPATH, './ancestor::div[contains(@class,"login") or contains(@class,"auth") or contains(@class,"signup")]')
                            continue  # Inside login form
                        except Exception:
                            pass
                        # NEVER click game-start buttons — chess.com's "Start Game"
                        # and quick-pair buttons are also .cc-button-primary, so
                        # clicking them here would auto-launch a game.
                        try:
                            _bt = (elem.text or "").strip().lower()
                        except Exception:
                            _bt = ""
                        if any(w in _bt for w in ("start game", "play", "find game",
                                                  "new game", "rematch", "create game",
                                                  "challenge", "accept")):
                            continue
                        elem.click()
                        dismissed += 1
                        time.sleep(0.5)
            except Exception:
                pass

        # Also look for "No thanks" / "Not now" / "Maybe later" buttons (upsell dismissal)
        try:
            for elem in self.driver.find_elements(By.CSS_SELECTOR, 'button, a, span'):
                try:
                    if not elem.is_displayed():
                        continue
                    text = elem.text.strip().lower()
                    if text in ['no thanks', 'not now', 'maybe later', 'dismiss',
                                'no, thanks', 'skip', 'close', 'x']:
                        elem.click()
                        dismissed += 1
                        time.sleep(0.3)
                        break
                except Exception:
                    pass
        except Exception:
            pass

        if dismissed:
            logging.info(f"Dismissed {dismissed} popups")

    def _tile_left(self):
        """Tile browser to the left half."""
        try:
            screen_w = self.driver.execute_script("return screen.width")
            screen_h = self.driver.execute_script("return screen.height")
            self.driver.set_window_rect(x=0, y=0, width=screen_w // 2, height=screen_h)
        except Exception:
            pass
        print("\033[90m  Tip: press Super+Left to snap browser to left half\033[0m")

    def _find_board(self):
        """Find the chess board element in the page."""
        from selenium.webdriver.common.by import By
        try:
            for selector in ['wc-chess-board', '.board', '#board-layout-chessboard',
                             '[id*="board"]', '.chess-board']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elems:
                    self.board_elem = elems[0]
                    return self.board_elem
        except Exception:
            pass
        return None

    def click_square(self, square, playing_color):
        """Click a chess square with human-like mouse movement.
        Uses Bezier curves, acceleration/deceleration, and natural overshoot."""
        from selenium.webdriver.common.action_chains import ActionChains
        import math

        board = self._find_board()
        if not board:
            logging.error("Selenium: board element not found")
            return False

        size = board.size
        bw, bh = size['width'], size['height']
        cell_w = bw / 8
        cell_h = bh / 8

        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if playing_color == "w":
            col = file
            row = 7 - rank
        else:
            col = 7 - file
            row = rank

        # Target with gaussian inaccuracy (humans miss center by ~8-15%)
        spread = cell_w * random.uniform(0.08, 0.15)
        target_x = col * cell_w + cell_w / 2 + random.gauss(0, spread)
        target_y = row * cell_h + cell_h / 2 + random.gauss(0, spread)

        # Offset from board center (ActionChains reference point)
        tx = target_x - bw / 2
        ty = target_y - bh / 2

        actions = ActionChains(self.driver)

        # Bezier curve control point — creates a natural arc
        # Humans don't move in straight lines, they curve slightly
        arc_strength = random.uniform(0.1, 0.3)
        ctrl_x = tx * 0.5 + random.gauss(0, abs(tx) * arc_strength + 5)
        ctrl_y = ty * 0.5 + random.gauss(0, abs(ty) * arc_strength + 5)

        # Number of steps — more for longer distances
        dist = math.sqrt(tx * tx + ty * ty)
        steps = max(3, min(8, int(dist / 30) + random.randint(2, 4)))

        for i in range(steps):
            # Ease-in-out timing (slow start, fast middle, slow end)
            t = (i + 1) / steps
            ease = t * t * (3 - 2 * t)  # Smoothstep

            # Quadratic Bezier: B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2
            # P0 = (0,0), P1 = control, P2 = target
            bx = 2 * (1 - ease) * ease * ctrl_x + ease * ease * tx
            by = 2 * (1 - ease) * ease * ctrl_y + ease * ease * ty

            # Add micro-jitter (hand tremor) — random, not perfectly linear
            tremor = random.uniform(0, 2.5) * (1 - ease * 0.7)
            bx += random.gauss(0, tremor)
            by += random.gauss(0, tremor)

            actions.move_to_element_with_offset(board, bx, by)

            # Variable timing — faster in middle, slower at start/end
            if i == 0:
                actions.pause(random.uniform(0.02, 0.06))  # Slow start
            elif i == steps - 1:
                actions.pause(random.uniform(0.01, 0.04))  # Approach
            else:
                actions.pause(random.uniform(0.005, 0.02))  # Fast middle

        # Small overshoot + correction (20% of the time)
        if random.random() < 0.2:
            overshoot = random.uniform(2, 6)
            ox = tx + random.choice([-1, 1]) * overshoot
            oy = ty + random.choice([-1, 1]) * overshoot
            actions.move_to_element_with_offset(board, ox, oy)
            actions.pause(random.uniform(0.03, 0.08))

        # Final position
        actions.move_to_element_with_offset(board, tx, ty)
        actions.pause(random.uniform(0.01, 0.06))
        actions.click()
        actions.perform()
        return True

    def execute_move(self, move, playing_color):
        """Execute a chess move by clicking source then destination square."""
        try:
            # Simulate network latency (50-200ms)
            time.sleep(random.uniform(0.05, 0.20))

            self.click_square(move.from_square, playing_color)
            # Human pause between source and destination click
            # Varies: fast drag-like (0.05s) to deliberate click-click (0.3s)
            time.sleep(random.choice([
                random.uniform(0.05, 0.12),   # Fast (drag-like)
                random.uniform(0.10, 0.25),   # Normal
                random.uniform(0.20, 0.40),   # Deliberate
            ]))
            self.click_square(move.to_square, playing_color)

            if move.promotion:
                time.sleep(0.3)
                # Chess.com shows promotion picker — queen is at the same spot
                if move.promotion == chess.QUEEN:
                    self.click_square(move.to_square, playing_color)
                elif move.promotion == chess.KNIGHT:
                    # Knight is one square below queen in the picker
                    from selenium.webdriver.common.action_chains import ActionChains
                    board = self._find_board()
                    if board:
                        cell_h = board.size['height'] / 8
                        ActionChains(self.driver).move_by_offset(0, cell_h).click().perform()

            logging.info(f"Selenium auto-moved: {move.uci()}")
            return True
        except Exception as e:
            logging.error(f"Selenium auto-move failed: {e}")
            return False

    def detect_game_over(self):
        """Check if chess.com is showing a game-over modal.
        Returns (is_over, result_text) e.g. (True, 'White Won by checkmate')."""
        from selenium.webdriver.common.by import By
        game_over_words = ['won', 'lost', 'draw', 'stalemate', 'timeout',
                           'resign', 'checkmate', 'time', 'abort', 'game over',
                           'game aborted', 'victory', 'defeat', 'abandoned',
                           'white won', 'black won', 'game review']
        try:
            # If we've landed on the post-game analysis / review page, the game is
            # over — chess.com auto-navigates there and it has no game-over modal to
            # find. Its URL carries /analysis or /review (a live game is /game/<id>
            # with neither), so marathon can recover and start the next game instead
            # of the bot sitting on the review board.
            _url = (self.driver.current_url or '').lower()
            if '/analysis' in _url or '/review' in _url:
                return True, self._get_game_result_text() or "Game ended"

            # Quick check: "Game Review" button is the most reliable signal
            # It ONLY appears after a game ends
            for sel in ['button', 'a', '[class*="review"]']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    try:
                        if elem.is_displayed() and 'game review' in elem.text.strip().lower():
                            # Found game review — get the result text from nearby elements
                            result = self._get_game_result_text()
                            return True, result or "Game ended"
                    except Exception:
                        pass

            # Check known modal selectors
            for sel in ['.game-over-modal', '.modal-game-over',
                        '[data-cy="game-over-modal"]', '.board-modal-container',
                        '.modal-container', '.board-modal-overlay',
                        '[class*="game-over"]', '[class*="modal"]']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    try:
                        if elem.is_displayed():
                            text = elem.text.strip()
                            if any(w in text.lower() for w in game_over_words):
                                return True, text.split('\n')[0]
                    except Exception:
                        pass

            # Check headings only (not all spans/divs — too many false positives)
            for sel in ['h1', 'h2', 'h3']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    try:
                        if not elem.is_displayed():
                            continue
                        text = elem.text.strip().lower()
                        if len(text) > 50:
                            continue
                        # Require more specific phrases, not just "won"
                        if any(w in text for w in ['white won', 'black won', 'checkmate',
                                                    'game aborted', 'game over',
                                                    'victory', 'defeat', 'you won']):
                            return True, elem.text.strip()
                    except Exception:
                        pass

            # Check for "New game" buttons
            for sel in ['button', 'a']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    try:
                        if elem.is_displayed():
                            text = elem.text.strip().lower()
                            if text.startswith('new ') and ('min' in text or 'game' in text):
                                return True, "Game ended"
                    except Exception:
                        pass
        except Exception:
            pass
        return False, ""

    def _get_game_result_text(self):
        """Extract the game result text from the page."""
        from selenium.webdriver.common.by import By
        try:
            for sel in ['h1', 'h2', 'h3', 'h4', '[class*="header"]',
                        '[class*="result"]', '[class*="title"]']:
                elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elems:
                    try:
                        if elem.is_displayed():
                            text = elem.text.strip()
                            if any(w in text.lower() for w in ['won', 'lost', 'draw',
                                                                'checkmate', 'time',
                                                                'resign', 'abort']):
                                return text
                    except Exception:
                        pass
        except Exception:
            pass
        return ""

    def select_time_control(self, tc):
        """Open the /play/online time dropdown and click the preset matching tc
        ('10|0', '15|10', '3 days', ...). Returns True if a preset was clicked.
        Flow (confirmed via DOM probe): the selector is a .time-selector dropdown
        button; presets show as 'N min', 'M + inc', or 'N day(s)'."""
        if not tc:
            return False
        from selenium.webdriver.common.by import By
        wanted = [w.lower() for w in time_control_labels(tc)]
        try:
            # 1) Open the dropdown (button currently shows e.g. "10 min (Rapid)")
            for dsel in ['.time-selector-next-component button',
                         '.cc-dropdown-button-component']:
                btns = self.driver.find_elements(By.CSS_SELECTOR, dsel)
                if btns and btns[0].is_displayed():
                    self.driver.execute_script("arguments[0].click();", btns[0])
                    time.sleep(1.0)
                    break

            # 2) Click the preset whose label matches (exact first, then substring)
            def _click_match(exact):
                for el in self.driver.find_elements(By.CSS_SELECTOR, 'button, [role="option"]'):
                    try:
                        if not el.is_displayed():
                            continue
                        txt = " ".join((el.text or "").split()).lower()
                        if not txt or 'start game' in txt or '(' in txt:
                            continue  # skip the current-selection header "10 min (Rapid)"
                        hit = (txt in wanted) if exact else any(w in txt for w in wanted)
                        if hit:
                            self.driver.execute_script("arguments[0].click();", el)
                            time.sleep(0.8)
                            logging.info(f"Time control '{tc}' set via '{txt}'")
                            print(f"\033[90m  Time control: {tc}\033[0m")
                            return True
                    except Exception:
                        pass
                return False

            if _click_match(True) or _click_match(False):
                return True
        except Exception as e:
            logging.error(f"select_time_control failed: {e}")
        logging.info(f"Time control '{tc}' not found in dropdown — leaving default")
        return False

    def click_start_game(self):
        """Click the green 'Start Game' quick-pair button. Returns True if clicked."""
        from selenium.webdriver.common.by import By
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, 'button.cc-button-primary, button'):
                try:
                    if el.is_displayed() and 'start game' in (el.text or '').strip().lower():
                        self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(2)
                        logging.info("Clicked Start Game")
                        return True
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"click_start_game failed: {e}")
        return False

    def setup_and_start_game(self):
        """Select the chosen time control (if any), then click Start Game."""
        self.select_time_control(SELECTED_TIME_CONTROL)
        time.sleep(0.5)
        return self.click_start_game()

    def start_new_game(self):
        """Start a new game. Navigates directly to play page."""
        try:
            # Use JS navigation (non-blocking, won't timeout)
            try:
                self.driver.execute_script("window.location.href = 'https://www.chess.com/play/online'")
            except Exception:
                # JS failed, try direct navigation with short timeout
                try:
                    self.driver.set_page_load_timeout(10)
                    self.driver.get('https://www.chess.com/play/online')
                except Exception:
                    try:
                        self.driver.execute_script("window.stop()")
                    except Exception:
                        pass
                finally:
                    try:
                        self.driver.set_page_load_timeout(120)
                    except Exception:
                        pass

            time.sleep(5)
            self.dismiss_popups()

            # Select the chosen time control (if any) and start the game
            if self.setup_and_start_game():
                return True

            # Fallback: click any play/start button
            from selenium.webdriver.common.by import By
            for attempt in range(3):
                for sel in ['button', 'a']:
                    for elem in self.driver.find_elements(By.CSS_SELECTOR, sel):
                        try:
                            if not elem.is_displayed():
                                continue
                            text = elem.text.strip().lower()
                            if any(w in text for w in ['play', 'start game', 'find game',
                                                        '10 min', '5 min', '3 min', '1 min',
                                                        'new game']):
                                if any(bad in text for bad in ['review', 'analysis']):
                                    continue
                                elem.click()
                                time.sleep(3)
                                self.dismiss_popups()
                                return True
                        except Exception:
                            pass
                time.sleep(2)
                self.dismiss_popups()

            logging.info("No play button found, assuming auto-queued")
            return True
        except Exception as e:
            logging.error(f"start_new_game failed: {e}")
            return True  # Return True anyway so marathon continues

    def is_alive(self):
        """Check if the browser is still open. Fast-fail by checking process first."""
        if self.driver is None:
            return False
        try:
            # Fast check: is the geckodriver process still running?
            if hasattr(self.driver, 'service') and self.driver.service.process:
                poll = self.driver.service.process.poll()
                if poll is not None:
                    return False  # Process exited
            # Slower check: can we talk to the browser?
            _ = self.driver.title
            return True
        except Exception:
            return False

    def is_game_live(self):
        """True once a real game is actually in progress.

        Signals verified against the real chess.com DOM (probed 2026-07-10, capturing
        the full lobby -> searching -> live transition):
          * LIVE loads at  /game/<id>  with a real opponent username and a 'Resign'
            control, and the board gets its 'flipped' class (only then is orientation
            correct for colour detection).
          * LOBBY sits at  /play/online  with a 'Start Game' button and an opponent
            literally named 'Opponent' (placeholder), and it renders BOTH clocks at a
            static time — so neither clocks NOR a non-empty opponent name mean 'live'.
        Fails OPEN (True) only on a DOM error so a hiccup can't deadlock the bot; the
        move-landed re-read in the main loop is the real safety net regardless."""
        try:
            url = (self.driver.current_url or '').lower()
            # /game/<id> is a LIVE game — but exclude the post-game analysis/review
            # page, whose URL is /analysis/game/live/<id>/review (contains /game/ but
            # is NOT playable; treating it as live sent the bot into "analysis mode").
            if '/game/' in url and '/analysis' not in url and '/review' not in url:
                return True
            # Still on /play/online etc. — require a positive in-game signal.
            return bool(self.driver.execute_script(r"""
                // A Resign / Abort control exists only in a live game.
                var rc = document.querySelector(
                    '[aria-label*="Resign" i],[aria-label*="Abort" i],[data-cy="resign"]');
                if (rc && rc.offsetParent !== null) return true;
                // A matched opponent's REAL username (the lobby shows the literal
                // placeholder "Opponent", so that doesn't count).
                var e = document.querySelector(
                    '.board-player-top [class*="username"],.player-top [class*="username"],'
                    + '.board-player-top a[href*="/member/"]');
                if (e && e.offsetParent !== null) {
                    var t = (e.textContent || '').trim().toLowerCase();
                    if (t && t !== 'opponent' && t.length > 1) return true;
                }
                return false;
            """))
        except Exception:
            return True

    def whose_turn(self):
        """Whose move it is, straight from chess.com's active-clock marker: the active
        player's clock carries the 'clock-player-turn' class, and the bottom clock is
        always ours (verified against the live DOM). Returns 'us', 'them', or None.
        This is the reliable signal for catching a tracking desync — if the DOM says
        it's our move but the bot thinks it's the opponent's, we missed their move."""
        try:
            r = self.driver.execute_script(r"""
                function vis(e){ return e && e.offsetParent !== null; }
                if (vis(document.querySelector('.clock-bottom.clock-player-turn'))) return 'us';
                if (vis(document.querySelector('.clock-top.clock-player-turn')))    return 'them';
                return '';
            """)
            return r if r in ('us', 'them') else None
        except Exception:
            return None

    def get_opponent_name(self):
        """Best-effort: read the opponent's (top player) username from the DOM."""
        from selenium.webdriver.common.by import By
        try:
            for sel in ['.board-player-top [data-test-element="user-tagline-username"]',
                        '.player-top .user-username-component',
                        '[data-test-element="user-tagline-username"]',
                        '.player-component .user-username-component',
                        '.board-player-top a[href*="/member/"]']:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed():
                        t = (el.text or "").strip()
                        if t:
                            return t
        except Exception:
            pass
        return None

    def report_opponent(self):
        """Best-effort: open chess.com's report/abuse dialog and submit a fair-play
        report. Returns True if it looks submitted, False otherwise (caller then
        shows manual instructions). NOTE: report DOM not yet verified — needs a probe."""
        from selenium.webdriver.common.by import By
        try:
            opened = False
            for sel in ['[aria-label*="report" i]', '[data-cy*="report"]',
                        'button[aria-label*="flag" i]', '.flag-component button',
                        '[class*="report"] button']:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed():
                        self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(1)
                        opened = True
                        break
                if opened:
                    break
            if not opened:
                return False
            # Choose a cheating / fair-play reason if a chooser appears
            for el in self.driver.find_elements(By.CSS_SELECTOR, 'button, [role="option"], label, li'):
                try:
                    t = (el.text or "").strip().lower()
                    if el.is_displayed() and any(w in t for w in ('cheat', 'fair play', 'engine', 'unfair')):
                        self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.5)
                        break
                except Exception:
                    pass
            # Submit
            for el in self.driver.find_elements(By.CSS_SELECTOR, 'button'):
                try:
                    t = (el.text or "").strip().lower()
                    if el.is_displayed() and t in ('report', 'submit', 'send report', 'confirm'):
                        self.driver.execute_script("arguments[0].click();", el)
                        time.sleep(1)
                        logging.info("Submitted opponent report")
                        return True
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"report_opponent failed: {e}")
        return False

    def close(self):
        """Close the browser and clean up geckodriver."""
        # Save cookies while the driver is still alive (get_cookies needs it) so a
        # by-hand login — incl. the one-time Turnstile solve — survives the next
        # launch. Replaces the old copy-cookies.sqlite-before-quit dance, which
        # copied the DB while Firefox still held it open (inconsistent / empty).
        try:
            if self.driver:
                self._save_cookies()
        except Exception as e:
            logging.error(f"Session persist failed: {e}")
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass
        # Kill any orphaned geckodriver processes
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'geckodriver' in proc.info['name'].lower():
                    proc.kill()
        except Exception:
            pass
        self.driver = None


def detect_playing_color(image, board_bounds):
    """Auto-detect which color we're playing by checking bottom two rows.
    Chess.com always shows your pieces at the bottom.
    Checks both row 7 (rank 1/8) and row 6 (rank 2/7) for robustness."""
    bx, by, bw, bh = board_bounds
    cell_w = bw / 8
    cell_h = bh / 8

    white_bottom = 0
    black_bottom = 0
    # Check bottom two rows (more robust than just one)
    for row in [7, 6]:
        for col in range(8):
            x1 = int(bx + col * cell_w)
            y1 = int(by + row * cell_h)
            cell = image[y1:y1+int(cell_h), x1:x1+int(cell_w)]
            color, ratio = classify_cell_color(cell)
            if color == "w":
                white_bottom += 1
            elif color == "b":
                black_bottom += 1

    # Also check top two rows to cross-validate
    white_top = 0
    black_top = 0
    for row in [0, 1]:
        for col in range(8):
            x1 = int(bx + col * cell_w)
            y1 = int(by + row * cell_h)
            cell = image[y1:y1+int(cell_h), x1:x1+int(cell_w)]
            color, ratio = classify_cell_color(cell)
            if color == "w":
                white_top += 1
            elif color == "b":
                black_top += 1

    # Bottom should have our pieces, top should have opponent's
    # If bottom=white and top=black → playing white
    # If bottom=black and top=white → playing black
    if white_bottom > black_bottom and black_top > white_top:
        return "w"
    elif black_bottom > white_bottom and white_top > black_top:
        return "b"
    # Single-side fallback
    if white_bottom > black_bottom + 2:
        return "w"
    elif black_bottom > white_bottom + 2:
        return "b"
    return None


def _detect_color_from_dom(ctrl):
    """Detect playing colour from chess.com's DOM. Returns 'w', 'b', or None.

    Uses board GEOMETRY rather than any class name (chess.com renames those):
      1. Each piece's class encodes its colour ('wp'..'wk' / 'bp'..'bk') and it has a
         real screen position. If White's pieces are drawn at the TOP of the board,
         we're looking from Black's side -> we're Black. (Primary — always available.)
      2. The 'square-FR' class encodes absolute file/rank; if higher ranks (Black's
         home) render at the top, it's White's view. (Backup.)
      3. Explicit 'flipped' class / orientation='black' attribute. (Last resort.)
    """
    try:
        # 1) Piece-colour geometry — which side's men sit at the top of the board?
        color = ctrl.driver.execute_script("""
            var b = document.querySelector('wc-chess-board')
                    || document.querySelector('chess-board') || document.querySelector('.board');
            if (!b) return '';
            var pieces = b.querySelectorAll('.piece');
            var wy = [], by = [];
            for (var i = 0; i < pieces.length; i++) {
                var cls = pieces[i].getAttribute('class') || '';
                var top = pieces[i].getBoundingClientRect().top;
                if (/(^|\\s)w[pnbrqk](\\s|$)/.test(cls)) wy.push(top);
                else if (/(^|\\s)b[pnbrqk](\\s|$)/.test(cls)) by.push(top);
            }
            if (!wy.length || !by.length) return '';
            var mean = function(a){ var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
            return mean(wy) < mean(by) ? 'b' : 'w';  // white men higher up => we view from Black
        """)
        if color in ('w', 'b'):
            return color
        # 2) square-FR rank geometry — higher rank drawn higher on screen => White's view
        color = ctrl.driver.execute_script("""
            var b = document.querySelector('wc-chess-board')
                    || document.querySelector('chess-board') || document.querySelector('.board');
            if (!b) return '';
            var pieces = b.querySelectorAll('.piece'), pts = [];
            for (var i = 0; i < pieces.length; i++) {
                var m = (pieces[i].getAttribute('class') || '').match(/square-(\\d)(\\d)/);
                if (m) pts.push([parseInt(m[2]), pieces[i].getBoundingClientRect().top]);
            }
            if (pts.length < 2) return '';
            pts.sort(function(a, b){ return a[0] - b[0]; });
            var lo = pts[0], hi = pts[pts.length - 1];
            if (lo[0] === hi[0]) return '';
            return (hi[1] < lo[1]) ? 'w' : 'b';
        """)
        if color in ('w', 'b'):
            return color
        # 3) Explicit flipped class / orientation attribute
        flip = ctrl.driver.execute_script("""
            var b = document.querySelector('wc-chess-board')
                    || document.querySelector('chess-board') || document.querySelector('.board');
            if (!b) return '';
            var cls = ((b.className||'') + ' ' + (b.getAttribute('class')||'')).toLowerCase();
            var orient = (b.getAttribute('orientation')||'').toLowerCase();
            if (cls.indexOf('flipped') !== -1 || orient === 'black') return 'b';
            if (orient === 'white') return 'w';
            return '';
        """)
        if flip in ('w', 'b'):
            return flip
    except Exception as e:
        logging.error(f"DOM color detection failed: {e}")
    return None


# Global Selenium controller (initialized in main loop if auto-move is enabled)
selenium_controller = None


def execute_move_on_screen(move, board_bounds, playing_color):
    """
    Execute a chess move. Uses Selenium if available (works on Wayland),
    falls back to pyautogui (X11 only).
    """
    global selenium_controller
    try:
        # Use Selenium if available
        if selenium_controller and selenium_controller.is_alive():
            return selenium_controller.execute_move(move, playing_color)

        # Fallback to pyautogui (X11 only)
        src_x, src_y = get_square_screen_coords(move.from_square, board_bounds, playing_color)
        dst_x, dst_y = get_square_screen_coords(move.to_square, board_bounds, playing_color)
        src_x += random.randint(-3, 3)
        src_y += random.randint(-3, 3)
        dst_x += random.randint(-3, 3)
        dst_y += random.randint(-3, 3)

        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        pyautogui.click(src_x, src_y)
        time.sleep(random.uniform(0.05, 0.15))
        pyautogui.click(dst_x, dst_y)
        if move.promotion:
            time.sleep(0.3)
            if move.promotion == chess.QUEEN:
                pyautogui.click(dst_x, dst_y)
            elif move.promotion == chess.KNIGHT:
                pyautogui.click(dst_x, dst_y + int(board_bounds[3] / 8))

        logging.info(f"Auto-moved: {move.uci()} ({src_x},{src_y}) → ({dst_x},{dst_y})")
        return True

    except Exception as e:
        logging.error(f"Auto-move failed: {e}")
        return False


# ─── FEN Persistence ──────────────────────────────────────────────────────────

def save_fen(fen, filepath="current_fen.txt"):
    with open(filepath, "w") as f:
        f.write(fen)


def load_fen(filepath="current_fen.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    return None


# ─── User Input ───────────────────────────────────────────────────────────────

# Standard Chess.com live time controls (minutes | increment-seconds), by category.
TIME_CONTROLS = [
    ("Bullet", ["1|0", "1|1", "2|1"]),
    ("Blitz",  ["3|0", "3|2", "5|0"]),
    ("Rapid",  ["10|0", "15|10", "30|0"]),
    ("Daily",  ["1 day", "3 days"]),
]

# Chosen time control (e.g. "10|0", "15|10", "3 days"). Set from the session in
# monitor_chessboard, read by the Selenium game-start helpers. None = pick in browser.
SELECTED_TIME_CONTROL = None

# Whether to auto-report opponents flagged as very likely engines. Set in
# monitor_chessboard from the session; read by maybe_report_opponent().
AUTO_REPORT = False


def normalize_time_control(s):
    """Normalize a user-typed time control to canonical form.
    Accepts '5', '5|0', '3|2', '3/2', '3+2', '1 day', '3days'. Returns str or None."""
    if not s:
        return None
    s = s.strip().lower()
    if "day" in s:
        num = "".join(ch for ch in s if ch.isdigit())
        if num:
            n = int(num)
            return f"{n} day" if n == 1 else f"{n} days"
        return None
    s = s.replace(" ", "")
    for sep in ("|", "/", "+"):
        if sep in s:
            a, _, b = s.partition(sep)
            if a.isdigit() and b.isdigit():
                return f"{int(a)}|{int(b)}"
            return None
    if s.isdigit():
        return f"{int(s)}|0"
    return None


def time_control_labels(tc):
    """Button labels Chess.com's time dropdown uses for a control.
    No increment -> 'N min'; with increment -> 'M + inc'; daily -> 'N day(s)'.
    e.g. '10|0'->['10 min']; '15|10'->['15 + 10', ...]; '3 days'->['3 days']."""
    if not tc:
        return []
    tc = tc.strip().lower()
    if "day" in tc:
        n = "".join(ch for ch in tc if ch.isdigit()) or "1"
        return [f"{n} days", f"{n} day"]
    if "|" in tc:
        m, _, inc = tc.partition("|")
        m, inc = m.strip(), inc.strip()
        if inc in ("0", ""):
            return [f"{m} min", f"{m} minutes"]
        return [f"{m} + {inc}", f"{m}+{inc}", f"{m}|{inc}", f"{m} min + {inc} sec"]
    return [tc]


def ask_time_control():
    """Prompt for a time control (preset or custom). Returns canonical string."""
    flat = []
    print("Time control:")
    for cat, presets in TIME_CONTROLS:
        start = len(flat)
        labels = "   ".join(f"[{start + i + 1}] {p}" for i, p in enumerate(presets))
        print(f"  {cat:<7} {labels}")
        flat.extend(presets)
    print("  Custom  [c] type your own, e.g. 7|3 or 2 days")
    while True:
        choice = input(f"Choose 1-{len(flat)} or c: ").strip().lower()
        if choice == "c":
            tc = normalize_time_control(input("Custom (minutes|increment): "))
            if tc:
                return tc
            print("Format like 5|0, 3|2, or 2 days.")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(flat):
            return flat[int(choice) - 1]
        print(f"Enter 1-{len(flat)} or c.")


def maybe_report_opponent(game, ctrl):
    """If auto-report is on and the opponent looks (strictly) like an engine,
    report them to chess.com. Strict threshold to minimize false positives —
    only fires when 'the system is sure'."""
    if not AUTO_REPORT or not game or not getattr(game, '_opp_cpls', None):
        return
    cpls = game._opp_cpls
    if len(cpls) < 10:
        return
    avg = sum(cpls) / len(cpls)
    best_pct = sum(1 for c in cpls if c < 10) / len(cpls) * 100
    if not (avg < 12 and best_pct > 80):
        return
    who = (ctrl.get_opponent_name() if ctrl and ctrl.is_alive() else None) or "opponent"
    print(f"\033[91m  ⚠ Auto-report: {who} looks like an engine "
          f"(avg CPL {avg:.0f}, {best_pct:.0f}% best over {len(cpls)} moves).\033[0m")
    if ctrl and ctrl.is_alive() and ctrl.report_opponent():
        print(f"\033[92m  Report submitted for {who}.\033[0m")
    else:
        print(f"\033[93m  Could not auto-open the report dialog — report {who} manually: "
              f"click their name → Report → Fair Play / Cheating.\033[0m")


def get_user_input():
    while True:
        try:
            skill_level = int(input("Skill level (0-20): ").strip())
            if 0 <= skill_level <= 20:
                break
        except ValueError:
            pass
        print("Enter 0-20.")

    while True:
        randomizer = input("Use anti-detection randomizer? (y/n): ").strip().lower()
        if randomizer in ("y", "n"):
            use_randomizer = randomizer == "y"
            break
        print("Enter 'y' or 'n'.")

    while True:
        auto = input("Auto-move pieces? (y/n): ").strip().lower()
        if auto in ("y", "n"):
            auto_move = auto == "y"
            break
        print("Enter 'y' or 'n'.")

    game_mode = None
    marathon = False
    if auto_move:
        while True:
            mode = input("Mode — bots/players/watch? (b/p/w): ").strip().lower()
            if mode in ("b", "p", "w"):
                if mode == "b":
                    game_mode = "bot"
                elif mode == "p":
                    game_mode = "player"
                else:
                    game_mode = "watch"
                break
            print("Enter 'b', 'p', or 'w'.")

        if game_mode != "watch":
            while True:
                m = input("Marathon mode — auto-start new games? (y/n): ").strip().lower()
                if m in ("y", "n"):
                    marathon = m == "y"
                    break
                print("Enter 'y' or 'n'.")

    time_control = None
    auto_report = False
    if auto_move and game_mode in ("bot", "player"):
        time_control = ask_time_control()
        while True:
            ar = input("Auto-report opponents that look like engines? (y/n): ").strip().lower()
            if ar in ("y", "n"):
                auto_report = ar == "y"
                break
            print("Enter 'y' or 'n'.")

    return skill_level, use_randomizer, auto_move, game_mode, marathon, time_control, auto_report


# ─── Main Loop ────────────────────────────────────────────────────────────────

stop_flag = threading.Event()
action_queue = queue.Queue()
paused = threading.Event()       # Set = paused
last_game = threading.Event()    # Set = finish current game then stop marathon


_terminal_settings = None

def _handle_hotkey(ch):
    """Apply a hotkey: P = pause/resume, L = last game (stop marathon after current)."""
    if ch == 'p':
        if paused.is_set():
            paused.clear()
            print("\n\033[92m▶ Resumed\033[0m")
        else:
            paused.set()
            print("\n\033[93m⏸ Paused (press P to resume)\033[0m")
    elif ch == 'l':
        if last_game.is_set():
            last_game.clear()
            print("\n\033[92m↻ Marathon continues\033[0m")
        else:
            last_game.set()
            print("\n\033[93m⏹ Last game — will stop after this game\033[0m")


def keyboard_listener():
    """Listen for hotkeys in a background thread (cross-platform)."""
    global _terminal_settings
    import sys

    # Windows: msvcrt (no termios/select on stdin)
    if platform.system() == "Windows":
        try:
            import msvcrt
        except Exception:
            return
        while not stop_flag.is_set():
            try:
                if msvcrt.kbhit():
                    _handle_hotkey(msvcrt.getwch().lower())
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)
        return

    # POSIX: cbreak terminal + select
    import select
    try:
        import tty, termios
        _terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except Exception:
        return
    try:
        while not stop_flag.is_set():
            if select.select([sys.stdin], [], [], 0.5)[0]:
                _handle_hotkey(sys.stdin.read(1).lower())
    except Exception:
        pass
    finally:
        restore_terminal()


def restore_terminal():
    """Restore terminal to normal mode."""
    global _terminal_settings
    if _terminal_settings is not None:
        try:
            import sys, termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _terminal_settings)
            _terminal_settings = None
        except Exception:
            pass


def signal_handler(sig, frame):
    stop_flag.set()
    restore_terminal()
    print("\nStopping...")
    global selenium_controller
    if selenium_controller:
        try:
            selenium_controller.close()
        except Exception:
            pass
    os._exit(0)


def _is_session_dead_error(exc):
    """True when an exception means the Firefox/geckodriver session is gone — i.e.
    we should restart the browser rather than keep retrying the call. Matches the
    connection-refused / invalid-session errors Selenium raises after a crash."""
    s = f"{type(exc).__name__} {exc}".lower()
    return any(k in s for k in (
        'invalid session id', 'invalidsessionid', 'no such window',
        'unable to connect', 'connection refused', 'newconnectionerror',
        'maxretryerror', 'remotedisconnected', 'failed to establish',
        'connection reset', 'session deleted', 'browser has closed',
        'tried to run command without establishing a connection',
    ))


def _recover_browser(old_ctrl, game_mode, playing_color, max_attempts=3):
    """Tear down a dead/broken Selenium session and bring a fresh one back up,
    reusing the saved chess.com session so we land logged in again. Relaunches
    with a new game so marathon play resumes — the game that was live when the
    browser died is unavoidably forfeited on time and can't be rejoined once it
    has flagged. Returns a live SeleniumController, or None if it can't recover
    after max_attempts."""
    try:
        if old_ctrl:
            old_ctrl.close()
    except Exception:
        pass
    for attempt in range(1, max_attempts + 1):
        try:
            ctrl = SeleniumController()
            ctrl.launch(game_mode=game_mode, playing_color=playing_color or "w")
            ctrl.ready = True
            return ctrl
        except Exception as e:
            logging.error(f"Browser relaunch attempt {attempt}/{max_attempts} failed: {e}")
            print(f"\033[93m  relaunch attempt {attempt}/{max_attempts} failed; retrying...\033[0m")
            time.sleep(min(4 * attempt, 15))
    return None


def _boards_equal(a, b):
    """Exact 8x8 equality of two raw detected boards."""
    try:
        return all(a[i][j] == b[i][j] for i in range(8) for j in range(8))
    except Exception:
        return False


def _is_start_position(raw_board):
    """True if the detected 8x8 board is a fresh chess starting position — the top two
    and bottom two ranks fully occupied and the middle four ranks empty (32 pieces,
    either orientation). Seeing this MID-game means a new game has begun (the previous
    one ended / the opponent abandoned and chess.com swapped the board in place)."""
    try:
        top = all(raw_board[0][j] and raw_board[1][j] for j in range(8))
        bottom = all(raw_board[6][j] and raw_board[7][j] for j in range(8))
        middle = all(not raw_board[r][j] for r in (2, 3, 4, 5) for j in range(8))
        return bool(top and bottom and middle)
    except Exception:
        return False


_BAD_READ_SAVES = 0


def _save_bad_read(tag, annotated, raw_board, fen=None):
    """Debug: when a board read looks wrong (impossible piece count / can't match /
    illegal), save the annotated overlay + the detected grid so the exact mis-
    classification can be inspected. Capped so it can't fill the disk."""
    global _BAD_READ_SAVES
    if _BAD_READ_SAVES >= 15:
        return
    try:
        _BAD_READ_SAVES += 1
        d = os.path.join(SCRIPT_DIR, "debug_frames")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"badread_{_BAD_READ_SAVES:02d}_{tag}")
        if annotated is not None:
            cv2.imwrite(base + ".png", annotated)
        with open(base + ".txt", "w") as f:
            f.write(f"reason: {tag}\n")
            if fen:
                f.write(f"tracked-FEN: {fen}\n")
            f.write("detected grid (as read, top row first; '.' = empty):\n")
            for row in raw_board:
                f.write(" ".join((c or ".") for c in row) + "\n")
        logging.info(f"Saved bad-read debug frame: {base}.png")
    except Exception as e:
        logging.error(f"_save_bad_read failed: {e}")


def _reread_raw_board(board_bounds, piece_templates, playing_color):
    """Capture the screen and extract the current raw 8x8 board (or None on failure).
    Used to confirm our auto-move actually registered on the real board before we
    commit to it internally."""
    try:
        path = capture_screenshot(None)
        img = cv2.imread(path)
        if img is None:
            return None
        board, _, _ = extract_board_from_image(img, piece_templates, board_bounds, playing_color)
        return board
    except Exception as e:
        logging.error(f"_reread_raw_board failed: {e}")
        return None


def _opponent_verdict(avg, best_pct):
    """Classify the opponent from average centipawn loss (+ best-move %). Returns
    (text, ansi_colour). The two engine-suspicion tiers depend on best%; the rest is a
    play-strength ladder with a rough rating ballpark (CPL→rating is only approximate —
    it drifts with time control and how forcing the game was)."""
    if avg < 15 and best_pct > 70:
        return f"⚠ LIKELY ENGINE — inhuman accuracy (avg CPL {avg:.0f}, {best_pct:.0f}% best)", "91"
    if avg < 25 and best_pct > 50:
        return f"⚠ SUSPICIOUS — very engine-like (avg CPL {avg:.0f}, {best_pct:.0f}% best)", "93"
    ladder = [
        (12,  "Near-flawless — master strength",      "~2200+"),
        (20,  "Expert level — very sharp",            "~2000"),
        (30,  "Strong club player",                   "~1800"),
        (42,  "Solid club player",                    "~1600"),
        (55,  "Decent intermediate",                  "~1400"),
        (70,  "Improving casual player",              "~1200"),
        (95,  "Casual player — the odd blunder",      "~1000"),
        (130, "Beginner — blunders regularly",        "~800"),
        (175, "Novice — hangs stuff often",           "~600"),
    ]
    for thresh, label, rating in ladder:
        if avg < thresh:
            return f"{label} (avg CPL {avg:.0f}, est. {rating})", "90"
    return f"Barely playing — drops pieces every few moves (avg CPL {avg:.0f}, est. <500)", "90"


def monitor_chessboard(playing_color, skill_level, use_randomizer, auto_move,
                       game_mode=None, marathon=False, time_control=None,
                       auto_report=False):
    """Main monitoring loop."""
    global SELECTED_TIME_CONTROL, AUTO_REPORT
    SELECTED_TIME_CONTROL = time_control
    AUTO_REPORT = auto_report
    engine = initialize_stockfish()
    current_skill = skill_level

    piece_templates = load_piece_templates()
    print(f"Loaded {len(piece_templates)} piece templates")

    board_bounds = None
    board_bounds_screen = None
    previous_screenshot_path = None
    cycle_count = 0
    uncertain_count = 0
    max_uncertain = 5
    detection_failures = 0
    invalid_frames = 0   # consecutive impossible-piece-count reads (phantom); escape hatch
    stats = {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'streak': 0, 'best_streak': 0}

    # Check if auto-move is available
    global selenium_controller
    if auto_move:
        try:
            selenium_controller = SeleniumController()
            # Pass playing_color=None for bot mode — it'll use default (white)
            selenium_controller.launch(game_mode=game_mode, playing_color=playing_color or "w")
            print("Auto-move: \033[92menabled (Selenium)\033[0m")
        except Exception as e:
            logging.error(f"Selenium launch failed: {e}")
            selenium_controller = None
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                print("Auto-move: \033[92menabled (pyautogui)\033[0m")
            except Exception as e2:
                print(f"Auto-move: \033[91mdisabled, suggesting moves only\033[0m")
                auto_move = False

    # GameState created after color detection (placeholder for now)
    game = None

    # Wait for Selenium to be ready (login, game setup, etc.)
    if selenium_controller and not selenium_controller.ready:
        while not stop_flag.is_set() and not selenium_controller.ready:
            time.sleep(1)

    print("\nWaiting for chessboard...")

    while not stop_flag.is_set():
        try:
            cycle_count += 1

            # Check if browser is still alive — recover if it crashed
            if selenium_controller and not selenium_controller.is_alive():
                print("\033[91mBrowser crashed — recovering...\033[0m")
                if game is not None and game.move_history:
                    print("\033[93m  (the in-progress game flags on time — starting fresh)\033[0m")
                    logging.info("Browser died mid-game; that game is forfeited on time.")
                new_ctrl = _recover_browser(selenium_controller, game_mode, playing_color)
                if new_ctrl is None:
                    print("\033[91mCould not bring the browser back after several tries — stopping.\033[0m")
                    selenium_controller = None
                    action_queue.put("game_over")
                    break
                selenium_controller = new_ctrl
                board_bounds = None
                board_bounds_screen = None
                playing_color = None
                game = None
                detection_failures = 0
                uncertain_count = 0
                print("\033[92mBrowser back up.\033[0m\nWaiting for chessboard...")
                time.sleep(3)
                continue

            # Check for game over FIRST every cycle (before board detection)
            if selenium_controller and selenium_controller.is_alive() and game is not None:
                _go, _gt = selenium_controller.detect_game_over()
                # Abort-at-start / vanished-game guard: if we're in a game but it's no
                # longer live (an abort drops us back to the lobby with no game-over
                # modal), treat it as over after a few confirmations so marathon queues a
                # fresh game instead of the move-retry loop grinding on forever.
                if not _go and game_mode == "player" and not selenium_controller.is_game_live():
                    game._not_live = getattr(game, '_not_live', 0) + 1
                    if game._not_live >= 3:
                        _go, _gt = True, "Game Aborted"
                        logging.info("Game no longer live (aborted/ended, no modal) — queuing next game")
                else:
                    game._not_live = 0
                if _go:
                    # Jump directly to game-over handling
                    result_text = _gt
                    game_ended = True
                    # Skip all detection/analysis — go straight to game-over section below
                    # (the game_ended check is at the end of the loop body)

                    # --- Inline game-over handling ---
                    rt = result_text.lower()
                    if 'abort' in rt:
                        outcome = "abort"
                    elif 'won' in rt or 'checkmate' in rt or 'victory' in rt:
                        if playing_color == "w" and 'white' in rt:
                            outcome = "win"
                        elif playing_color == "b" and 'black' in rt:
                            outcome = "win"
                        elif playing_color == "w" and 'black' in rt:
                            outcome = "loss"
                        elif playing_color == "b" and 'white' in rt:
                            outcome = "loss"
                        else:
                            outcome = "win"  # Generic "won" — assume us
                    elif 'draw' in rt or 'stalemate' in rt:
                        outcome = "draw"
                    elif 'lost' in rt or 'timeout' in rt or 'resign' in rt or 'defeat' in rt:
                        outcome = "loss"
                    elif 'abandon' in rt:
                        outcome = "abort"
                    else:
                        outcome = "unknown"

                    if outcome == "abort":
                        print(f"\n\033[93mGame aborted.\033[0m {result_text}")
                    else:
                        stats['games'] += 1
                        if outcome == "win":
                            stats['wins'] += 1
                            stats['streak'] += 1
                            stats['best_streak'] = max(stats['best_streak'], stats['streak'])
                            print(f"\n\033[1m\033[92mWin!\033[0m {result_text}")
                        elif outcome == "loss":
                            stats['losses'] += 1
                            stats['streak'] = 0
                            print(f"\n\033[1m\033[91mLoss.\033[0m {result_text}")
                        elif outcome == "draw":
                            stats['draws'] += 1
                            print(f"\n\033[1m\033[93mDraw.\033[0m {result_text}")
                        else:
                            print(f"\n\033[1mGame Over:\033[0m {result_text}")

                    # Game summary
                    move_count = len(game.move_history) if game else 0
                    print(f"\033[90m  ─── Game Summary ({move_count} moves) ───\033[0m")
                    if game and hasattr(game, '_opp_cpls') and game._opp_cpls:
                        cpls = game._opp_cpls
                        avg = sum(cpls) / len(cpls)
                        best_pct = sum(1 for c in cpls if c < 10) / len(cpls) * 100
                        excellent_pct = sum(1 for c in cpls if c < 20) / len(cpls) * 100
                        inaccuracies = sum(1 for c in cpls if 50 <= c < 100)
                        mistakes = sum(1 for c in cpls if 100 <= c < 200)
                        blunders = sum(1 for c in cpls if c >= 200)
                        worst = max(cpls)
                        print(f"\033[90m  Opponent: avg CPL={avg:.0f}, best={best_pct:.0f}%, excellent={excellent_pct:.0f}%\033[0m")
                        print(f"\033[90m  Opponent: {inaccuracies} inaccuracies, {mistakes} mistakes, {blunders} blunders (worst={worst} CPL)\033[0m")
                        if len(cpls) >= 8:
                            _v, _c = _opponent_verdict(avg, best_pct)
                            print(f"\033[{_c}m  VERDICT: {_v}\033[0m")
                    maybe_report_opponent(game, selenium_controller)
                    print(f"\033[90m  ──────────────────────────\033[0m")
                    if stats['games'] > 0:
                        print(f"\033[90m  Record: {stats['wins']}W-{stats['losses']}L-{stats['draws']}D "
                              f"({stats['games']} games, streak: {stats['streak']}, best: {stats['best_streak']})\033[0m")

                    if marathon and not last_game.is_set() and selenium_controller.is_alive():
                        print("\033[90m  Marathon: queuing new game...\033[0m")
                        if selenium_controller.start_new_game():
                            board_bounds = None
                            board_bounds_screen = None
                            detection_failures = 0
                            uncertain_count = 0
                            playing_color = None
                            game = None
                            print("\nWaiting for chessboard...")
                            continue
                    if last_game.is_set():
                        print("\033[93mMarathon stopped (last game flag).\033[0m")
                    action_queue.put("game_over")
                    break

            # Periodically dismiss chess.com popups (only when ready)
            if selenium_controller and selenium_controller.ready and cycle_count % 5 == 1:
                try:
                    selenium_controller.dismiss_popups()
                except Exception:
                    pass

            # Always capture full screen for reliable detection
            screenshot_path = capture_screenshot(None)
            if not os.path.exists(screenshot_path):
                continue

            # Wait for animation to settle
            if previous_screenshot_path:
                for _ in range(config["move_stabilization_attempts"]):
                    if not are_images_similar(previous_screenshot_path, screenshot_path):
                        time.sleep(config["move_stabilization_delay"])
                        screenshot_path = capture_screenshot(None)
                    else:
                        break
            previous_screenshot_path = screenshot_path

            full_image = cv2.imread(screenshot_path)
            if full_image is None:
                continue

            # Detect board location every frame (fast enough, handles resizing)
            detected = find_board_by_colors(full_image)
            if detected is None:
                detection_failures += 1
                if detection_failures <= 3:
                    time.sleep(1)
                else:
                    # Extended failure — slow down to avoid spinning
                    if detection_failures % 10 == 0:
                        print("Waiting for chessboard...")
                    time.sleep(3)
                continue

            bx, by, bw, bh = detected

            # Board position stability: if we have an established position,
            # reject detections that jump too far (likely false positives)
            if board_bounds is not None and detection_failures < 3:
                old_bx, old_by, old_bw, old_bh = board_bounds
                dx = abs(bx - old_bx)
                dy = abs(by - old_by)
                dw = abs(bw - old_bw)
                if dx > 150 or dy > 150 or dw > 200:
                    # Suspiciously large jump — keep old position
                    logging.info(f"Rejected board jump: ({bx},{by},{bw}) vs ({old_bx},{old_by},{old_bw})")
                    continue

            # Report board detection (first time, or if size/position changed significantly)
            if board_bounds is None:
                # Wait for the game to actually be LIVE before detecting colour or
                # moving. Otherwise, as White, the loop fires move 1 into the
                # matchmaking / countdown screen — and reads colour off the pre-game
                # board (which flips when you're assigned Black) — then desyncs and
                # hangs. Only online play has this race; bot/watch start instantly.
                # Bounded wait (~25s) so a DOM-selector miss degrades to old behaviour
                # instead of deadlocking.
                if (game_mode == "player" and game is None
                        and selenium_controller and selenium_controller.is_alive()):
                    if selenium_controller.is_game_live():
                        selenium_controller._live_wait = 0
                    else:
                        waited = getattr(selenium_controller, '_live_wait', 0) + 1
                        selenium_controller._live_wait = waited
                        if waited <= 25:
                            if waited == 1:
                                print("\033[90m  Waiting for the game to start...\033[0m")
                            logging.info("Board detected but game not live yet — waiting")
                            time.sleep(1.0)
                            continue
                        logging.info("Game-live wait timed out — proceeding anyway")
                        selenium_controller._live_wait = 0

                print(f"\033[92mBoard found:\033[0m {bw}x{bh} at ({bx},{by}), "
                      f"cell={bw//8}x{bh//8}px")

                # Auto-detect playing color — poll until DOM and CV agree
                if playing_color is None:
                    print("\033[90mDetecting playing color...\033[0m")
                    for attempt in range(8):
                        if attempt > 0:
                            time.sleep(1.5)
                            fresh_path = capture_screenshot(None)
                            fresh_img = cv2.imread(fresh_path)
                            if fresh_img is not None:
                                full_image = fresh_img
                                new_det = find_board_by_colors(full_image)
                                if new_det:
                                    detected = new_det

                        dom_color = None
                        cv_color = None

                        if selenium_controller and selenium_controller.is_alive():
                            dom_color = _detect_color_from_dom(selenium_controller)
                        cv_color = detect_playing_color(full_image, detected)

                        # DOM board orientation is the ground truth — trust it first.
                        # (The CV colour read can misjudge which side is at the bottom,
                        # especially when black pieces are hard to classify — that's the
                        # "detected White when I was Black" bug.)
                        if dom_color:
                            playing_color = dom_color
                            break
                        # Fall back to the CV read only if the DOM can't tell us yet
                        if cv_color and attempt >= 2:
                            playing_color = cv_color
                            break

                    if playing_color is None:
                        playing_color = "w"
                        print(f"\033[93mCould not detect color, assuming White\033[0m")
                    else:
                        color_name = "White" if playing_color == "w" else "Black"
                        print(f"\033[92mPlaying as:\033[0m {color_name} (auto-detected)")

                # Initialize game state now that we know the color
                if game is None:
                    game = GameState(playing_color)
                    game_num = stats['games'] + 1
                    logging.info(f"=== NEW GAME #{game_num} as {playing_color} ===")
                    print(f"\033[90m  Game #{game_num} started\033[0m")
            elif detection_failures > 3:
                print(f"\033[92mBoard re-found:\033[0m {bw}x{bh} at ({bx},{by})")
            elif abs(bw - board_bounds[2]) > 20 or abs(bx - board_bounds[0]) > 30:
                print(f"\033[93mBoard moved/resized:\033[0m {bw}x{bh} at ({bx},{by}), "
                      f"cell={bw//8}x{bh//8}px")

            detection_failures = 0
            board_bounds = detected
            board_bounds_screen = detected

            cell_size = min(bw, bh) // 8
            use_color_only = cell_size < 45
            if use_color_only and cycle_count <= 2:
                print(f"\033[93mSmall cells ({cell_size}px):\033[0m using color-only detection + move matching")

            # Skip if game state not initialized yet (waiting for color detection)
            if game is None:
                continue

            # Extract board state
            current_board, annotated, piece_count = extract_board_from_image(
                full_image, piece_templates, board_bounds, playing_color,
                color_only=use_color_only
            )
            cv2.imwrite(os.path.join(SCRIPT_DIR, "annotated_chessboard.png"), annotated)

            # Sanity: skip if piece count is impossible (>32 can't happen — it's a
            # phantom read, usually a last-move highlight counted as a piece). If it
            # PERSISTS, the board region / detection is stuck (force_sync rejects it,
            # nothing matches, and the desync guard just re-syncs to the same bad
            # board) — so re-detect everything from scratch rather than freezing on a
            # won game like it did before.
            if piece_count > 32 or piece_count < 2:
                logging.info(f"Skipping frame: invalid piece count {piece_count}")
                # Only capture a MILD over-count (a mid-game phantom worth diagnosing) —
                # skip the huge startup VS-splash counts (~40+), which are known/handled.
                if 32 < piece_count <= 36:
                    _save_bad_read(f"count{piece_count}", annotated, current_board)
                invalid_frames += 1
                if invalid_frames >= 15:
                    logging.info("Persistent invalid board read — re-detecting board + game from scratch")
                    print("\033[93m  Board read stuck (phantom piece) — re-detecting from scratch...\033[0m")
                    invalid_frames = 0
                    game = None
                    playing_color = None
                    board_bounds = None
                    board_bounds_screen = None
                    detection_failures = 0
                    uncertain_count = 0
                time.sleep(1)
                continue
            invalid_frames = 0   # a valid read — clear the stuck counter

            # New-game detection: if we're mid-game but the board is a fresh starting
            # position, the previous game ended (e.g. the opponent abandoned) and a new
            # one has begun — chess.com swaps the board in place, so game-over never
            # fired and we'd otherwise get stuck rejecting the 32-piece board forever.
            # Reset and re-detect the new game.
            if (game is not None and game.move_history
                    and _is_start_position(current_board)):
                logging.info("Fresh starting position mid-game — new game began; resetting")
                print("\033[92m  New game detected — switching to it.\033[0m")
                game = None
                playing_color = None
                board_bounds = None
                board_bounds_screen = None
                detection_failures = 0
                uncertain_count = 0
                continue

            # Update game state — use color-only matching if in that mode
            if use_color_only:
                result = game.update_from_detection_color_only(current_board)
            else:
                result = game.update_from_detection(current_board)

                # If template matching gave uncertain result, retry with color-only
                if result == "uncertain":
                    color_board, _, _ = extract_board_from_image(
                        full_image, piece_templates, board_bounds, playing_color,
                        color_only=True
                    )
                    color_result = game.update_from_detection_color_only(color_board)
                    if color_result != "uncertain":
                        result = color_result
                        logging.info(f"Color-only fallback succeeded: {result}")
                    else:
                        # Genuinely couldn't read this frame — grab it for diagnosis.
                        _save_bad_read("nomatch", annotated, current_board,
                                       game.fen if game else None)

            # Recovery-play: if we've been re-synced to "our turn" (by the desync guard or a
            # post-reset seed) but the detected board is unchanged simply because WE haven't
            # moved yet, nothing below triggers our move — the play path only fires on a
            # freshly-detected OPPONENT move — so we sit idle and flag our own clock on a
            # position we already hold (this is what just threw a won +5 game). Promote a
            # STABLE our-turn read to a play, but only once the DOM clock confirms it really
            # is our move, so we never move on a bad turn inference. After we move the play
            # block sets is_our_turn=False, so this won't re-fire (except the move-didn't-land
            # retry, which is intended).
            if (result == "no_change" and game and game.is_our_turn
                    and engine is not None and auto_move and board_bounds_screen
                    and selenium_controller and selenium_controller.is_alive()
                    and getattr(game, '_no_change_count', 0) >= 2
                    and selenium_controller.whose_turn() == 'us'):
                logging.info("Recovery-play: our turn but board idle — playing instead of stalling")
                print("\033[93m  Our move (recovered) — playing...\033[0m")
                result = "our_turn"

            if result == "no_change":
                no_change_count = getattr(game, '_no_change_count', 0) + 1 if game else 0
                if game:
                    game._no_change_count = no_change_count

                # Desync guard (reliable): we think we're waiting for the opponent, but
                # if chess.com's active-clock says it's OUR move, we missed their move —
                # re-sync and play instead of sitting here until we flag on time (this is
                # what threw the +20 game). Requires the DOM to say 'our turn' for several
                # consecutive reads so a normal just-moved lag (detection catching up)
                # doesn't trip it. Uses the whose_turn() clock marker, not a clock-delta.
                if (game and not game.is_our_turn and no_change_count >= 2
                        and selenium_controller and selenium_controller.is_alive()):
                    if selenium_controller.whose_turn() == 'us':
                        game._turn_desync = getattr(game, '_turn_desync', 0) + 1
                    else:
                        game._turn_desync = 0
                    if getattr(game, '_turn_desync', 0) >= 4:
                        game._turn_desync = 0
                        # Only recover if we can sync to a LEGAL board. If the read is
                        # garbled (force_sync rejects it), do NOT flip to our-turn or act
                        # on it — that corrupts the game and is exactly what hung + flagged
                        # winning games. Leave state and wait for a clean read.
                        if game.force_sync(current_board):
                            logging.info("Active-clock says it's our move but we were waiting — re-synced")
                            print("\033[93m  Missed a move (clock shows it's our turn) — re-syncing...\033[0m")
                            game.is_our_turn = True
                            game._skip_sync_cycles = 0
                            game._no_change_count = 0
                            uncertain_count = 0
                            continue

                # Show alive indicator every 10 cycles (~5s)
                if no_change_count % 10 == 0 and game and not game.is_our_turn:
                    clock = get_clock_remaining(selenium_controller)
                    clock_str = f" (clock: {clock}s)" if clock else ""
                    print(f"\033[90m  ...waiting for opponent{clock_str}\033[0m")

                # Watchdog: if stuck 8+ cycles (~4s), force comparison
                # DON'T clear last_raw_board — that allows phantom pieces to sync
                # Instead, just reset the counter so we keep trying
                if game and no_change_count >= 8:
                    game._no_change_count = 0
                    # Force a fresh board comparison by marking board as changed
                    game.consecutive_same = 0
                    logging.info("Watchdog: resetting detection counters")
                    # Stall-escape: if we've been stuck through many watchdog cycles
                    # (~3 min of no progress — a garbled read we can neither sync nor
                    # match, not a normal wait), nuke and re-detect board+colour+game
                    # from scratch rather than stalling until the game abandons.
                    game._stall_watchdogs = getattr(game, '_stall_watchdogs', 0) + 1
                    if game._stall_watchdogs >= 6:
                        logging.info("Stalled too long — re-detecting board + game from scratch")
                        print("\033[93m  Stalled (can't read the board cleanly) — re-detecting from scratch...\033[0m")
                        game = None
                        playing_color = None
                        board_bounds = None
                        board_bounds_screen = None
                        detection_failures = 0
                        uncertain_count = 0
                        time.sleep(1)
                        continue

                # Check for game over every 3 cycles (~2.5s)
                if selenium_controller and no_change_count % 3 == 0:
                    game_ended, result_text = selenium_controller.detect_game_over()
                    if game_ended:
                        if game:
                            game._no_change_count = 0
                        # Reuse the game-over handling by not continuing
                    else:
                        time.sleep(config["loop_delay"])
                        continue
                else:
                    time.sleep(config["loop_delay"])
                    continue

            if result == "uncertain":
                uncertain_count += 1
                # Force sync faster when auto-moving (board changes rapidly)
                sync_threshold = 2 if (auto_move and selenium_controller) else max_uncertain
                if uncertain_count >= sync_threshold:
                    game.force_sync(current_board)
                    uncertain_count = 0
                    logging.info("Force-synced board state")
                time.sleep(0.5)
                continue

            if game:
                game._no_change_count = 0
                game._stall_watchdogs = 0   # real progress — clear the stall counter

            uncertain_count = 0
            save_fen(game.fen)

            # Pause check
            while paused.is_set() and not stop_flag.is_set():
                time.sleep(0.5)

            if result == "our_turn" and engine is not None:
                # Opponent's move was detected — re-enable sync for future use
                game._skip_sync_cycles = 0

                # Sanity: don't play if piece count is way off (bad detection)
                if game.piece_count > 32:
                    logging.warning(f"Skipping move: invalid piece count {game.piece_count}")
                    game.force_sync([["" for _ in range(8)] for _ in range(8)])
                    time.sleep(1)
                    continue

                # It's our turn — opponent just moved (or game just started)
                if game.move_history:
                    last = game.move_history[-1]

                    # Analyze opponent's move quality (skip if low on time)
                    opp_eval = ""
                    clock = get_clock_remaining(selenium_controller)
                    try:
                        if clock is None or clock > 60:
                            # Quick single analysis of pre-move position
                            pre_board = game.board.copy()
                            pre_board.pop()
                            opp_analysis = engine.analyse(
                                pre_board, chess.engine.Limit(depth=6), multipv=1)
                            opp_best = opp_analysis[0]["pv"][0]
                            opp_best_score = opp_analysis[0]["score"].relative.score() or 0
                            # Use our already-available current eval instead of separate analysis
                            post_score = -(opp_best_score)  # Approximate
                            cpl = 0 if last == opp_best else abs(opp_best_score) // 2
                            played_best = (last == opp_best)
                        else:
                            raise Exception("Low time — skip opp analysis")

                        if played_best:
                            opp_eval = f" \033[91m[BEST move, CPL=0]\033[0m"
                        elif cpl < 20:
                            opp_eval = f" \033[91m[excellent, CPL={cpl}]\033[0m"
                        elif cpl < 50:
                            opp_eval = f" \033[93m[good, CPL={cpl}]\033[0m"
                        elif cpl < 100:
                            opp_eval = f" \033[92m[inaccuracy, CPL={cpl}]\033[0m"
                        elif cpl < 200:
                            opp_eval = f" \033[92m[mistake, CPL={cpl}]\033[0m"
                        else:
                            opp_eval = f" \033[92m[blunder! CPL={cpl}]\033[0m"

                        # Track opponent's overall accuracy
                        if not hasattr(game, '_opp_cpls'):
                            game._opp_cpls = []
                        game._opp_cpls.append(cpl)
                        avg_cpl = sum(game._opp_cpls) / len(game._opp_cpls)
                        best_count = sum(1 for c in game._opp_cpls if c < 10)
                        best_pct = best_count / len(game._opp_cpls) * 100

                        # Sus detection
                        sus = ""
                        if len(game._opp_cpls) >= 5:
                            if avg_cpl < 15 and best_pct > 70:
                                sus = " \033[91m⚠ LIKELY ENGINE\033[0m"
                            elif avg_cpl < 25 and best_pct > 50:
                                sus = " \033[93m⚠ suspicious\033[0m"

                        opp_eval += f"\033[90m  avg CPL={avg_cpl:.0f}, best%={best_pct:.0f}%{sus}\033[0m"
                    except Exception as e:
                        logging.error(f"Opponent analysis failed: {e}")

                    print(f"\033[90mOpponent played: {last.uci()}{opp_eval}\033[0m")

                # Try opening book first (instant, no engine, undetectable)
                book_move = get_book_move(game.board)
                opp_cpl = None
                if hasattr(game, '_opp_cpls') and game._opp_cpls:
                    opp_cpl = sum(game._opp_cpls) / len(game._opp_cpls)

                if book_move and use_randomizer:
                    move = book_move
                    best_score = 30  # Approximate neutral eval
                    sel_score = 30
                    logging.info(f"Book move: {move.uci()}")
                else:
                    move, best_score, sel_score = get_best_move(
                        engine, game.board, game.piece_count,
                        use_randomizer, current_skill, opp_avg_cpl=opp_cpl,
                    )

                if move is not None:
                    game._last_suggested = move

                    # Human-like think time based on position complexity + clock
                    if use_randomizer:
                        clock = get_clock_remaining(selenium_controller)
                        move_num = len(game.move_history) // 2 + 1
                        delay = calculate_human_think_time(
                            game.board, move, best_score, sel_score,
                            move_num, game.piece_count, clock)
                        clock_str = f", clock={clock}s" if clock else ""
                        print(f"\033[90mThinking {delay:.1f}s (move {move_num}{clock_str})...\033[0m")
                        time.sleep(delay)

                    print_status(game, move, best_score)

                    # AC intel
                    diff = abs(best_score - sel_score)
                    move_num = len(game.move_history) // 2 + 1
                    blunder_zone = 0.03 + max(0, (move_num - 25) * 0.005)
                    if use_randomizer:
                        parts = []
                        if diff > 0:
                            parts.append(f"sub-optimal (gap={diff/100:.1f})")
                        else:
                            parts.append("best move")
                        parts.append(f"eval: {format_eval(best_score)}\033[90m→{format_eval(sel_score)}")
                        parts.append(f"\033[90mskill={current_skill}")
                        if opp_cpl:
                            parts.append(f"opp CPL={opp_cpl:.0f}")
                        parts.append(f"blunder%={blunder_zone:.0%}")
                        print(f"\033[90m  AC: {', '.join(parts)}\033[0m")

                    if auto_move and board_bounds_screen:
                        if execute_move_on_screen(move, board_bounds_screen, playing_color):
                            time.sleep(1.0)   # let the move animation settle
                            # Verify the move actually landed before committing. If the
                            # board is UNCHANGED from just before we "moved", the click
                            # went nowhere (game not live yet / click missed) — so DON'T
                            # pretend we moved (that's the "thinks it moved when it hasn't"
                            # hang). We reject ONLY on an exact match to the pre-move board,
                            # so a real move (which changes the board) is never rejected.
                            pre_move = game.last_raw_board
                            fresh = _reread_raw_board(board_bounds, piece_templates, playing_color)
                            if (fresh is not None and pre_move is not None
                                    and _boards_equal(fresh, pre_move)):
                                logging.warning(f"Move {move.uci()} didn't register "
                                                f"(board unchanged) — not committing")
                                print("\033[93m  Move didn't register (game not live?) — retrying...\033[0m")
                                time.sleep(1.0)
                            else:
                                game.board.push(move)
                                game.move_history.append(move)
                                game.is_our_turn = False
                                # DON'T clear last_raw_board — this prevents phantom piece
                                # sync. Skip FEN sync for the rest of the game.
                                game._skip_sync_cycles = 9999
                                logging.info(f"Auto-played: {move.uci()}")
                        else:
                            logging.error("Auto-move failed, showing move instead")

                    current_skill = adjust_skill_level(
                        engine, current_skill, game.piece_count, diff, game,
                        our_eval=best_score,
                    )
                else:
                    logging.error("Failed to get best move")

            elif result == "their_turn":
                # We just moved — only notify if it differs from suggestion
                if game.move_history and hasattr(game, '_last_suggested'):
                    last = game.move_history[-1]
                    if last != game._last_suggested:
                        print(f"\033[93mYou played: {last.uci()} (suggested: {game._last_suggested.uci()})\033[0m")

            # Check for game end — Selenium modal detection (most reliable)
            game_ended = False
            result_text = ""
            if selenium_controller and selenium_controller.is_alive():
                game_ended, result_text = selenium_controller.detect_game_over()

            # Also check python-chess board state
            if not game_ended and game.board.is_game_over():
                game_ended = True
                result_text = game.board.result()

            # Also check for win visually (only after enough moves)
            if not game_ended and game and len(game.move_history) >= 4:
                if check_win(full_image):
                    game_ended = True
                    result_text = "You won!"

            if game_ended:
                # Parse result
                rt = result_text.lower()
                if 'abort' in rt:
                    outcome = "abort"
                elif 'won' in rt or 'checkmate' in rt or 'you won' in rt or 'victory' in rt:
                    if playing_color == "w" and 'white' in rt:
                        outcome = "win"
                    elif playing_color == "b" and 'black' in rt:
                        outcome = "win"
                    elif 'you won' in rt or 'victory' in rt:
                        outcome = "win"
                    else:
                        outcome = "loss"
                elif 'draw' in rt or 'stalemate' in rt:
                    outcome = "draw"
                elif 'lost' in rt or 'timeout' in rt or 'time' in rt or 'resign' in rt or 'defeat' in rt:
                    outcome = "loss"
                elif 'abandon' in rt:
                    outcome = "abort"
                else:
                    outcome = "unknown"

                # Update streak
                if outcome == "abort":
                    print(f"\n\033[93mGame aborted.\033[0m {result_text}")
                    # Don't count aborts in stats
                else:
                    stats['games'] += 1
                    if outcome == "win":
                        stats['wins'] += 1
                        stats['streak'] += 1
                        stats['best_streak'] = max(stats['best_streak'], stats['streak'])
                        print(f"\n\033[1m\033[92mWin!\033[0m {result_text}")
                    elif outcome == "loss":
                        stats['losses'] += 1
                        stats['streak'] = 0
                        print(f"\n\033[1m\033[91mLoss.\033[0m {result_text}")
                    elif outcome == "draw":
                        stats['draws'] += 1
                        print(f"\n\033[1m\033[93mDraw.\033[0m {result_text}")
                    else:
                        print(f"\n\033[1mGame Over:\033[0m {result_text}")

                # Game summary
                move_count = len(game.move_history) if game else 0
                print(f"\033[90m  ─── Game Summary ({move_count} moves) ───\033[0m")

                # Our performance
                if game and hasattr(game, '_last_suggested'):
                    our_moves = move_count // 2
                    print(f"\033[90m  Our play: {our_moves} moves, skill={current_skill}\033[0m")

                # Opponent analysis
                if game and hasattr(game, '_opp_cpls') and game._opp_cpls:
                    cpls = game._opp_cpls
                    avg = sum(cpls) / len(cpls)
                    best_pct = sum(1 for c in cpls if c < 10) / len(cpls) * 100
                    excellent_pct = sum(1 for c in cpls if c < 20) / len(cpls) * 100
                    inaccuracies = sum(1 for c in cpls if 50 <= c < 100)
                    mistakes = sum(1 for c in cpls if 100 <= c < 200)
                    blunders = sum(1 for c in cpls if c >= 200)
                    worst = max(cpls)

                    print(f"\033[90m  Opponent: avg CPL={avg:.0f}, "
                          f"best={best_pct:.0f}%, excellent={excellent_pct:.0f}%\033[0m")
                    print(f"\033[90m  Opponent: {inaccuracies} inaccuracies, "
                          f"{mistakes} mistakes, {blunders} blunders (worst={worst} CPL)\033[0m")

                    # Verdict
                    if len(cpls) >= 8:
                        _v, _c = _opponent_verdict(avg, best_pct)
                        print(f"\033[{_c}m  VERDICT: {_v}\033[0m")
                    else:
                        print(f"\033[90m  VERDICT: Too few moves to judge ({len(cpls)} analyzed)\033[0m")

                maybe_report_opponent(game, selenium_controller)
                print(f"\033[90m  ──────────────────────────\033[0m")

                if stats['games'] > 0:
                    print(f"\033[90m  Record: {stats['wins']}W-{stats['losses']}L-{stats['draws']}D "
                          f"({stats['games']} games, streak: {stats['streak']}, "
                          f"best: {stats['best_streak']})\033[0m")

                if marathon and not last_game.is_set() and selenium_controller and selenium_controller.is_alive():
                    print("\033[90m  Marathon: queuing new game...\033[0m")
                    if selenium_controller.start_new_game():
                        board_bounds = None
                        board_bounds_screen = None
                        detection_failures = 0
                        uncertain_count = 0
                        playing_color = None
                        game = None
                        print("\nWaiting for chessboard...")
                        continue
                    else:
                        print("Could not start new game automatically.")

                if last_game.is_set():
                    print("\033[93mMarathon stopped (last game flag).\033[0m")
                action_queue.put("game_over")
                break

            time.sleep(config["loop_delay"])

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            # Recover if the browser died — either it reads as dead, or the error that
            # just fired is itself a dead-session/connection error from a call that
            # raced the crash (kills the retry-storm-then-hang path).
            if selenium_controller and (not selenium_controller.is_alive()
                                        or _is_session_dead_error(e)):
                print("\033[91mBrowser crashed — recovering...\033[0m")
                new_ctrl = _recover_browser(selenium_controller, game_mode, playing_color)
                if new_ctrl is None:
                    print("\033[91mCould not bring the browser back after several tries — stopping.\033[0m")
                    selenium_controller = None
                    action_queue.put("game_over")
                    break
                selenium_controller = new_ctrl
                board_bounds = None
                board_bounds_screen = None
                playing_color = None
                game = None
                detection_failures = 0
                uncertain_count = 0
                print("\033[92mBrowser back up.\033[0m\nWaiting for chessboard...")
                time.sleep(3)
            else:
                time.sleep(2)

    if not stop_flag.is_set():
        action_queue.put("finished")


def handle_user_input(playing_color, skill_level, use_randomizer, auto_move):
    """Handle post-game user interaction."""
    while True:
        action = action_queue.get()
        if action in ("won", "game_over"):
            while True:
                choice = input("Play again? (y/n): ").strip().lower()
                if choice == "y":
                    stop_flag.clear()
                    t = threading.Thread(
                        target=monitor_chessboard,
                        args=(playing_color, skill_level, use_randomizer, auto_move),
                    )
                    t.start()
                    break
                elif choice == "n":
                    print("Thanks for playing!")
                    os._exit(0)
                else:
                    print("Enter 'y' or 'n'.")
        elif action == "finished":
            break


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # Windows: enable ANSI colour codes in the console (VT processing)
    if platform.system() == "Windows":
        try:
            import ctypes
            _k = ctypes.windll.kernel32
            _k.SetConsoleMode(_k.GetStdHandle(-11), 7)
        except Exception:
            pass

    import sys
    print("\033[1m═══ Chess.com Assistant ═══\033[0m\n")

    if '--setup' in sys.argv:
        run_setup()
        sys.exit(0)

    # Notify (only) if a newer, signature-verified version is on GitHub
    try:
        check_for_update()
    except Exception:
        pass

    # Check for -r (resume) flag
    if '-r' in sys.argv or '--resume' in sys.argv:
        session = load_session()
        if session:
            skill_level = session['skill_level']
            use_randomizer = session['use_randomizer']
            auto_move = session['auto_move']
            game_mode = session.get('game_mode')
            marathon = session.get('marathon', False)
            time_control = session.get('time_control')
            auto_report = session.get('auto_report', False)
            print(f"\033[92mResuming:\033[0m skill={skill_level}, AC={'on' if use_randomizer else 'off'}, "
                  f"auto={'on' if auto_move else 'off'}, mode={game_mode or 'manual'}, "
                  f"marathon={'on' if marathon else 'off'}, time={time_control or 'manual'}, "
                  f"report={'on' if auto_report else 'off'}")
        else:
            print("\033[93mNo previous session found. Starting fresh.\033[0m")
            skill_level, use_randomizer, auto_move, game_mode, marathon, time_control, auto_report = get_user_input()
    else:
        skill_level, use_randomizer, auto_move, game_mode, marathon, time_control, auto_report = get_user_input()
    print()

    # Save session for future -r
    save_session(skill_level, use_randomizer, auto_move, game_mode, marathon, time_control, auto_report)
    save_config(config)

    if auto_move:
        print("\033[90mHotkeys: P = pause/resume, L = last game (stop marathon after current)\033[0m")

    # Start keyboard listener for hotkeys
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    # playing_color will be auto-detected from the board
    monitor_thread = threading.Thread(
        target=monitor_chessboard,
        args=(None, skill_level, use_randomizer, auto_move, game_mode, marathon, time_control, auto_report),
    )
    monitor_thread.start()

    input_thread = threading.Thread(
        target=handle_user_input,
        args=(None, skill_level, use_randomizer, auto_move),
    )
    input_thread.start()

    try:
        monitor_thread.join()
        input_thread.join()
    finally:
        restore_terminal()
        if selenium_controller:
            selenium_controller.close()
