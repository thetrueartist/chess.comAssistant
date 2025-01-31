# ChessVisionBot ♟️👁️🤖

**A smart chess bot that combines computer vision and AI to play chess in real-time on Chess.com

## Features ✨
- 🖥️ **Screen Analysis**: Automatically detects chessboard position using OpenCV
- 🤖 **AI Integration**: Powered by Stockfish engine with dynamic difficulty adjustment (0-20 skill levels)
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

*WORK IN PROGRESS, A LITTLE BUGGY :p*
