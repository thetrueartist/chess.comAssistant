# ChessVisionBot ♟️👁️🤖

**A smart chess bot that combines computer vision and AI to play chess in real-time on Chess.com**

## Features ✨
- 🖥️ **Screen Analysis**: Automatically detects chessboard position using OpenCV + PyTorch Vision

- 🤖 **ML Chess Engine Integration With Stockfish**: Powered by Stockfish engine with dynamic difficulty adjustment (0-20 skill levels)

- 🕵️ **Anti-Detection System**: 
  - Randomized move timing patterns
  - Human-like move selection variance
  - Dynamic skill adjustment

- 🧠 **Smart Game Analysis**:
  - Real-time FEN position recognition
  - Automatic game phase detection (opening/middlegame/endgame)
  - Adaptive thinking time based on position complexity

- 📊 **Performance Monitoring**:
  - CPU/Memory usage tracking
  - Detailed error logging
  - Move validation system

- ♟️ **Game Features**:
  - Supports both white and black pieces
  - Win condition detection
  - Multi-game session support

## How It Works 🔧
1. **Computer Vision Pipeline**:
   - Screenshot capture with `pyscreenshot`
   - Chessboard detection using OpenCV contour analysis
   - Piece recognition with ResNet50 deep learning model
   - Perspective transformation for board alignment

2. **AI Engine**:
   - Stockfish integration with optimized parameters
   - Dynamic depth adjustment (15-30 plies)
   - Position complexity analysis using move entropy
   - Multi-variant move selection with weighted randomness

3. **Anti-Detection System**:
   - Randomized delay patterns (0.1-2.5s)
   - Skill level variability during gameplay
   - Natural move timing based on position complexity

## Requirements 📦
```bash
Python 3.8+
Stockfish 15+ (path configuration required)
torch
torchvision
opencv-python
pyscreenshot
python-chess
psutil
```
## Installation 🛠️
```bash
pip install opencv-python numpy pyscreenshot torch torchvision python-chess psutil pillow

(you might need to use --break-system-packages on newer versions of linux for that pip command)

sudo apt-get install stockfish  # Debian/Ubuntu
brew install stockfish          # MacOS
```

## NOTES 🗒️

<h3> NOW WITH WINDOWS SUPPORT! </h3> 
<h4> :) </h4>

![chessAssistantWindows](https://github.com/thetrueartist/chess.comAssistant/blob/main/chessAssistantWINDOWS.png)

**Chess.com Assistant is ideally used with the terminal on the right half of your screen and the chess.com window on the left half. It additionally only works on the default Chess.com theme.**

**It works best at 1440p resolution.**
![chessAssistantLinux](https://github.com/thetrueartist/chess.comAssistant/blob/main/chessAssistant.png)

*WORK IN PROGRESS, A LITTLE BUGGY :p*



