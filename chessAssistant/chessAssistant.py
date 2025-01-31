import os
import cv2
import numpy as np
import pyscreenshot as ImageGrab
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import chess
import chess.engine
import threading
import time
import signal
import psutil
import queue
import logging
import random

# Configure logging
logging.basicConfig(
    filename="chess_engine_error.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_user_input():
    while True:
        playing_color = (
            input("Are you playing as white or black? (w/b): ").strip().lower()
        )
        if playing_color in ("w", "b"):
            break
        print("Invalid input. Please enter 'w' for white or 'b' for black.")

    while True:
        try:
            skill_level = int(input("Enter the skill level (0-20): ").strip())
            if 0 <= skill_level <= 20:
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter a valid skill level between 0-20.")

    while True:
        use_randomizer = (
            input("Do you want to use a randomizer to avoid detection? (y/n): ")
            .strip()
            .lower()
        )
        if use_randomizer in ("y", "n"):
            use_randomizer = use_randomizer == "y"
            break
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")

    return playing_color, skill_level, use_randomizer


playing_color, skill_level, use_randomizer = get_user_input()

# Set default values for threads and hash size
threads = max(5, os.cpu_count() // 2)  # Reduce the number of threads
hash_size = 8192  # Increase the hash size to 8 GB

# Capture the screen region where the chessboard is displayed
bbox = (100, 100, 900, 900)  # Example bounding box

# Define a flag for stopping the background thread
stop_flag = threading.Event()

# Queue for communication between threads
action_queue = queue.Queue()


def signal_handler(sig, frame):
    global stop_flag
    stop_flag.set()
    print("\nGracefully stopping...")
    os._exit(0)


signal.signal(signal.SIGINT, signal_handler)


def initialize_stockfish():
    stockfish_path = (
        "/usr/games/stockfish"  # Update this path to where Stockfish is located
    )
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": hash_size})
    return engine


def move_to_readable(move, board):
    piece = board.piece_at(move.from_square)
    piece_name = piece.symbol().upper() if piece else ""
    from_square = chess.SQUARE_NAMES[move.from_square]
    to_square = chess.SQUARE_NAMES[move.to_square]
    piece_map = {
        "P": "pawn",
        "N": "knight",
        "B": "bishop",
        "R": "rook",
        "Q": "queen",
        "K": "king",
    }
    return f"\033[1m\033[92mMove {piece_map.get(piece_name, '')} from {from_square} to {to_square}\033[0m"


def capture_screenshot(bbox):
    screenshot_path = "/tmp/chessboard.png"
    screenshot = ImageGrab.grab(bbox).convert("RGB")
    screenshot.save(screenshot_path)
    return screenshot_path


def detect_chessboard(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        chessboard_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(chessboard_contour, True)
        approx = cv2.approxPolyDP(chessboard_contour, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array(
                [
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1],
                ],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped, maxWidth // 8, maxHeight // 8
    return None, None, None


def preprocess_board_image(image):
    # Contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)
    return denoised


def match_template(piece_image, board_image, threshold=0.7):
    # Multi-scale template matching
    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)
    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    scales = np.linspace(0.8, 1.2, 5)
    matches = []

    for scale in scales:
        resized = cv2.resize(piece_gray, None, fx=scale, fy=scale)
        res = cv2.matchTemplate(board_gray, resized, cv2.TM_CCOEFF_NORMED)
        locations = np.where(res >= threshold)
        for loc in zip(*locations[::-1]):
            matches.append((loc, scale, res[loc[1]][loc[0]]))

    # Non-maximum suppression
    matches.sort(key=lambda x: x[2], reverse=True)
    filtered_matches = []
    for m1 in matches:
        should_add = True
        for m2 in filtered_matches:
            dist = np.sqrt((m1[0][0] - m2[0][0]) ** 2 + (m1[0][1] - m2[0][1]) ** 2)
            if dist < 20:  # Adjust threshold as needed
                should_add = False
                break
        if should_add:
            filtered_matches.append(m1)

    return [m[0] for m in filtered_matches]


def validate_board_state(board, prev_board=None):
    # Basic chess rules validation
    piece_counts = {"P": 0, "p": 0, "K": 0, "k": 0}
    for row in board:
        for piece in row:
            if piece in piece_counts:
                piece_counts[piece] += 1

    # Validate piece counts
    if piece_counts["K"] != 1 or piece_counts["k"] != 1:
        return False
    if piece_counts["P"] > 8 or piece_counts["p"] > 8:
        return False

    # Validate move if previous board exists
    if prev_board is not None:
        changes = sum(
            1 for i in range(8) for j in range(8) if board[i][j] != prev_board[i][j]
        )
        if changes > 4:  # Max changes for castling
            return False

    return True


def detect_movement(prev_image, curr_image, threshold=0.1):
    diff = cv2.absdiff(prev_image, curr_image)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours of movement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and analyze movement patterns
    valid_movements = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # Adjust thresholds based on board size
            x, y, w, h = cv2.boundingRect(cnt)
            valid_movements.append((x + w / 2, y + h / 2))

    return len(valid_movements) > 0, valid_movements


def extract_board_from_image(
    board_image, piece_templates, cell_width, cell_height, playing_color
):
    board = np.full((8, 8), "", dtype=str)
    piece_count = 0  # Initialize piece count
    for piece, template in piece_templates.items():
        if template is not None:
            positions = match_template(template, board_image, threshold=0.7)
            for pos in positions:
                x, y = pos
                row = y // cell_height
                col = x // cell_width
                piece_type = piece[1] if piece[0] == "b" else piece[1].upper()
                if playing_color == "b":
                    row, col = 7 - row, 7 - col  # Flip the board for black
                if board[row, col] == "":
                    piece_count += 1
                board[row, col] = piece_type
                cv2.rectangle(
                    board_image,
                    (x, y),
                    (x + template.shape[1], y + template.shape[0]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    board_image,
                    piece_type,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return board, board_image, piece_count  # Return piece count as well


def board_to_fen(board):
    fen = ""
    for row in board:
        empty_count = 0
        for cell in row:
            if cell == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += cell
        if empty_count > 0:
            fen += str(empty_count)
        fen += "/"
    fen = fen[:-1]
    if playing_color == "b":
        fen += " b KQkq - 0 1"
    else:
        fen += " w KQkq - 0 1"
    return fen


def get_dynamic_move_time(piece_count, use_randomizer):
    """
    Determines the dynamic move time based on the stage of the game.
    """
    if use_randomizer:
        if random.random() < 0.1:
            return random.uniform(0.1, 1.0)
        else:
            return random.uniform(0.1, 0.5)
    else:
        game_stage = get_game_stage(piece_count)
        print(f"Game stage detected: {game_stage}")  # Debug print
        logging.info(f"Game stage detected: {game_stage}")
        if game_stage == "opening":
            return 0.3  # 1.5 seconds in the opening
        elif game_stage == "middlegame":
            return 0.5  # 2.5 seconds in the middlegame
        else:  # endgame
            return 0.8  # 4 seconds in the endgame


def get_game_stage(piece_count):
    """
    Determines the stage of the game: opening, middlegame, or endgame.
    """
    logging.debug(f"Piece count: {piece_count}")
    print(f"Debug - Piece count: {piece_count}")  # Debug print

    # Adjust thresholds as needed based on your understanding of the game phases
    if piece_count > 25:
        return "opening"
    elif 14 < piece_count <= 25:
        return "middlegame"
    else:
        return "endgame"


def get_move_entropy(moves):
    """Calculate position complexity to vary play strength"""
    scores = [move["score"].relative.score() for move in moves]
    score_range = max(scores) - min(scores)
    return score_range / 100  # Normalized complexity score


def get_best_move_with_dynamic_time(
    engine,
    fen,
    piece_count,
    use_randomizer,
    skill_level,
    opponent_rating=None,
    retries=3,
):
    """
    Determines the best move dynamically while maintaining anti-detection mechanics.
    Enhanced to handle None scores and late-game scenarios better.
    """
    for attempt in range(retries):
        try:
            board = chess.Board(fen)

            # Adjust base depth based on piece count for late game
            if piece_count <= 16:  # Late game
                base_depth = random.randint(18, 22)
                min_depth = 18
            elif piece_count <= 25:  # Middle game
                base_depth = random.randint(15, 22)
                min_depth = 15
            else:  # Opening
                base_depth = random.randint(15, 20)
                min_depth = 15

            entropy = 0

            # Force higher depth in critical positions
            try:
                quick_analysis = engine.analyse(
                    board, chess.engine.Limit(depth=10), multipv=3
                )
                top_scores = []
                for move in quick_analysis:
                    score = move["score"].relative.score()
                    if score is not None:
                        top_scores.append(score)

                if top_scores:  # Only calculate if we have valid scores
                    score_diff = max(abs(score) for score in top_scores)
                    if score_diff > 150:  # Critical position
                        base_depth = max(base_depth, 20)
            except Exception as e:
                logging.error(f"Quick analysis failed: {e}")

            # Calculate entropy with safe score handling
            try:
                quick_moves = engine.analyse(
                    board, chess.engine.Limit(depth=8), multipv=3
                )
                valid_scores = []
                for move in quick_moves:
                    score = move["score"].relative.score()
                    if score is not None:
                        valid_scores.append(score)

                if valid_scores:
                    score_range = max(valid_scores) - min(valid_scores)
                    entropy = score_range / 100
                else:
                    entropy = 0.3  # Default to moderate complexity
            except Exception as e:
                logging.error(f"Entropy calculation failed: {e}")
                entropy = 0.3

            depth = max(
                base_depth + (int(entropy * 1) if entropy > 0.3 else 0), min_depth
            )

            # Add this line to limit the maximum depth to 30
            depth = min(depth, 30)  # Limit depth to a maximum of 30

            print(f"Selected depth for this move: {depth}")
            logging.info(f"Selected depth for this move: {depth}")

            start_time = time.time()

            try:
                analysis = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=5,
                    options={"Skill Level": skill_level},
                )
            except Exception as e:
                logging.error(f"Depth-based analysis failed: {e}")
                time_limit = 2.0 if piece_count <= 16 else 1.0
                analysis = engine.analyse(
                    board,
                    chess.engine.Limit(time=time_limit),
                    multipv=5,
                    options={"Skill Level": skill_level},
                )

            end_time = time.time()
            actual_thinking_time = end_time - start_time

            print(f"Actual thinking time: {actual_thinking_time:.2f} seconds")
            logging.info(f"Actual thinking time: {actual_thinking_time:.2f} seconds")

            if not analysis:
                raise Exception("No analysis returned")

            best_move = analysis[0]["pv"][0]
            best_move_score = analysis[0]["score"].relative.score()

            # Handle None scores
            if best_move_score is None:
                best_move_score = 0  # Default to neutral position if score is None

            if not best_move or not board.is_legal(best_move):
                raise Exception("Invalid move generated")

            print(f"Best move: {best_move}, Score: {best_move_score}")
            logging.info(f"Best move: {best_move}, Score: {best_move_score}")

            position_advantage = (
                best_move_score / 100 if best_move_score is not None else 0
            )
            if abs(position_advantage) > 1.5:
                max_diff = 80 + abs(position_advantage) * 8
            else:
                max_diff = 40 + entropy * 15

            print(f"Max difference threshold: {max_diff}")
            logging.info(f"Max difference threshold: {max_diff}")

            if opponent_rating:
                adjustment_factor = max(1, (opponent_rating / 2000))
                if opponent_rating < 1800:
                    adjustment_factor *= 0.8
                elif opponent_rating > 2200:
                    adjustment_factor *= 1.2
                max_diff *= adjustment_factor

            valid_moves = []
            for move in analysis:
                move_score = move["score"].relative.score()
                if move_score is None:
                    move_score = 0  # Default to neutral position
                score_diff = abs(move_score - best_move_score)
                if score_diff <= max_diff:
                    valid_moves.append((move, score_diff))

            print(f"Valid moves count: {len(valid_moves)}")
            logging.info(f"Valid moves count: {len(valid_moves)}")

            if not valid_moves:
                return best_move, best_move_score, best_move_score

            move_weights = []
            for move, diff in valid_moves:
                weight = (max_diff - diff) / max_diff
                weight *= random.uniform(0.8, 1.2)
                move_weights.append(weight)

            print(f"Move weights: {move_weights}")
            logging.info(f"Move weights: {move_weights}")

            if random.random() < 0.1 and abs(position_advantage) < 1:
                selected_move = random.choice(valid_moves)[0]
                selected_score = selected_move["score"].relative.score()
                if selected_score is None:
                    selected_score = 0
                print(
                    f"Selected move: {selected_move['pv'][0]}, Score: {selected_score}"
                )
                print(f"Move rank: {analysis.index(selected_move) + 1}")
                logging.info(
                    f"Selected move: {selected_move['pv'][0]}, Score: {selected_score}"
                )
                logging.info(f"Move rank: {analysis.index(selected_move) + 1}")
                return (
                    selected_move["pv"][0],
                    best_move_score,
                    selected_score,
                )

            selected_move = random.choices(
                [m[0] for m in valid_moves], weights=move_weights, k=1
            )[0]
            selected_score = selected_move["score"].relative.score()
            if selected_score is None:
                selected_score = 0

            print(f"Selected move: {selected_move['pv'][0]}, Score: {selected_score}")
            print(f"Move rank: {analysis.index(selected_move) + 1}")
            logging.info(
                f"Selected move: {selected_move['pv'][0]}, Score: {selected_score}"
            )
            logging.info(f"Move rank: {analysis.index(selected_move) + 1}")

            # Calculate and display variable move time
            base_time = random.uniform(1.5, 2.5)
            complexity_factor = 1 + (min(entropy, 0.5) * 0.4)
            position_factor = 1 + (min(abs(position_advantage), 2) * 0.15)
            move_time = base_time * complexity_factor * position_factor
            print(f"Variable move time: {move_time:.2f} seconds")
            logging.info(f"Variable move time: {move_time:.2f} seconds")

            if use_randomizer:
                time.sleep(random.uniform(0.1, 0.5))

            return (
                selected_move["pv"][0],
                best_move_score,
                selected_score,
            )

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} - An error occurred: {e}")
            if attempt < retries - 1:
                try:
                    engine = initialize_stockfish()
                except Exception as e2:
                    logging.error(f"Failed to reinitialize engine: {e2}")
            time.sleep(0.5)

    try:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            emergency_move = legal_moves[0]
            logging.warning(f"Using emergency move: {emergency_move}")
            return emergency_move, 0, 0  # Return neutral scores for emergency moves
    except Exception as e:
        logging.error(f"Emergency move selection failed: {e}")

    return None, None, None


def are_images_similar(image1_path, image2_path, threshold=0.1):
    """Enhanced similarity check that's more tolerant of moving pieces"""
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        return False
    if image1.shape != image2.shape:
        return False

    # Calculate both absolute difference and structural similarity
    diff = cv2.absdiff(image1, image2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size

    # Calculate movement area
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If we detect exactly one or two contoured areas (piece movement + possible highlight)
    # and they're reasonable size for a piece, consider this a moving piece
    is_piece_moving = False
    if 1 <= len(contours) <= 2:
        total_area = sum(cv2.contourArea(c) for c in contours)
        avg_piece_area = (cell_width * cell_height) * 0.8  # Expected piece size
        if total_area < avg_piece_area * 2:  # Allow for piece + trail
            is_piece_moving = True

    return (non_zero_count / total_pixels) < threshold or is_piece_moving


def validate_move(prev_board, new_board, max_piece_diff=2):
    """
    Validates if the new board state is physically possible given the previous state.
    Returns True if the move appears valid, False otherwise.
    """
    if prev_board is None:
        return True

    # Count differences between boards
    differences = 0
    moved_from = []
    moved_to = []

    for i in range(8):
        for j in range(8):
            if prev_board[i][j] != new_board[i][j]:
                differences += 1
                if prev_board[i][j] != "" and new_board[i][j] == "":
                    moved_from.append((i, j))
                elif prev_board[i][j] == "" and new_board[i][j] != "":
                    moved_to.append((i, j))

    # Basic validation checks
    if differences > max_piece_diff:  # Too many pieces changed
        return False
    if len(moved_from) != len(moved_to):  # Unequal number of source/destination changes
        return False
    if differences == 0:  # No changes detected
        return False

    # For each source-destination pair, verify it's a valid chess move
    for src in moved_from:
        for dst in moved_to:
            # Calculate Manhattan distance
            distance = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
            piece_type = prev_board[src[0]][src[1]].upper()

            # Validate based on piece type
            if piece_type == "P":  # Pawn
                if distance > 2:  # Pawns can't move more than 2 squares
                    continue
            elif piece_type in ["N"]:  # Knight
                if distance != 3:  # Knights must move in L-shape
                    continue
            elif piece_type in ["B", "R", "Q", "K"]:  # Other pieces
                if (
                    piece_type == "K" and distance > 2
                ):  # King can't move more than 1 square (2 for castling)
                    continue
            return True  # Found at least one valid move

    return False


def check_win(board_image):
    win_images = ["trophy.png", "crown.png"]
    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    for win_image_path in win_images:
        win_image = cv2.imread(win_image_path, cv2.IMREAD_UNCHANGED)
        if win_image is not None:
            win_gray = cv2.cvtColor(win_image, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(board_gray, win_gray, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            if len(loc[0]) > 0:
                return True
    return False


def save_fen(fen, filepath="current_fen.txt"):
    with open(filepath, "w") as f:
        f.write(fen)


def load_fen(filepath="current_fen.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    return None


def adjust_skill_level(engine, current_skill, piece_count, move_score_difference):
    """
    Adjust the skill level dynamically based on the game state and move quality.
    Includes mechanisms to prevent downward spirals and allow recovery.
    """
    new_skill = current_skill

    # Minimum skill level threshold
    min_skill_level = 5

    # Adjust based on game phase and move quality
    if piece_count > 25:  # Opening phase
        new_skill = min(current_skill + 1, 20)  # Increase skill slightly in opening
    elif piece_count <= 14:  # Endgame
        new_skill = max(
            current_skill - 1, min_skill_level
        )  # Decrease skill in endgame, but not below threshold
    elif move_score_difference > 200:  # Significant mistake
        new_skill = max(
            current_skill - 1, min_skill_level
        )  # Penalize significant mistakes, limit decrease
    else:
        new_skill = min(current_skill + 1, 20)  # Reward good moves

    # Introduce some randomness in skill level adjustments to simulate human variability
    if random.random() < 0.2:  # 20% chance to adjust skill level randomly
        adjustment = random.choice([-1, 0, 1])
        new_skill = max(min(new_skill + adjustment, 20), min_skill_level)

    if new_skill != current_skill:
        engine.configure({"Skill Level": new_skill})
        logging.info(f"Skill level adjusted: {new_skill}")

    return new_skill


def monitor_chessboard():
    engine = initialize_stockfish()
    current_skill = skill_level  # Initialize the skill level

    piece_templates = {
        "br": cv2.imread("black_rook.png"),
        "bn": cv2.imread("black_knight.png"),
        "bb": cv2.imread("black_bishop.png"),
        "bq": cv2.imread("black_queen.png"),
        "bk": cv2.imread("black_king.png"),
        "bp": cv2.imread("black_pawn.png"),
        "wr1": cv2.imread("white_rook.png"),
        "wr2": cv2.imread("white_rook_withwhitebackground.png"),
        "wn1": cv2.imread("white_knight.png"),
        "wn2": cv2.imread("white_knight_withgreenbackground.png"),
        "wb1": cv2.imread("white_bishop.png"),
        "wb2": cv2.imread("white_bishop_withgreenbackground.png"),
        "wq1": cv2.imread("white_queen.png"),
        "wq2": cv2.imread("white_queen_withgreenbackground.png"),
        "wk1": cv2.imread("white_king.png"),
        "wk2": cv2.imread("white_king_withwhitebackground.png"),
        "wp1": cv2.imread("white_pawn.png"),
        "wp2": cv2.imread("white_pawn_withgreenbackground.png"),
    }

    board = chess.Board()  # Initialize the chess board
    player_turn = playing_color == "w"  # True if it's the player's turn
    previous_fen = load_fen()
    previous_screenshot_path = None
    previous_board = None
    consecutive_invalid_moves = 0
    max_invalid_moves = 3
    last_valid_board = None

    while not stop_flag.is_set():
        try:
            screenshot_path = capture_screenshot(bbox)
            if not os.path.exists(screenshot_path):
                logging.error(f"Screenshot file not found: {screenshot_path}")
                continue

            # Wait for piece movement to complete
            if previous_screenshot_path:
                movement_stabilization_attempts = 0
                while movement_stabilization_attempts < 5:
                    if not are_images_similar(
                        previous_screenshot_path, screenshot_path
                    ):
                        time.sleep(0.2)  # Wait a short time for movement to complete
                        screenshot_path = capture_screenshot(bbox)
                        movement_stabilization_attempts += 1
                    else:
                        break

            previous_screenshot_path = screenshot_path

            board_image, cell_width, cell_height = detect_chessboard(screenshot_path)
            if board_image is None:
                print("Chessboard not detected, retrying in 5 seconds...")
                logging.error("Chessboard not detected, retrying in 5 seconds...")
                time.sleep(5)
                continue

            (
                current_board,
                annotated_board_image,
                piece_count,
            ) = extract_board_from_image(
                board_image, piece_templates, cell_width, cell_height, playing_color
            )
            cv2.imwrite("annotated_chessboard.png", annotated_board_image)

            # Validate the move
            if not validate_move(previous_board, current_board):
                consecutive_invalid_moves += 1
                if consecutive_invalid_moves >= max_invalid_moves:
                    logging.warning(
                        "Multiple invalid moves detected, resetting to last valid board state"
                    )
                    consecutive_invalid_moves = 0
                    current_board = (
                        last_valid_board.copy()
                        if last_valid_board is not None
                        else current_board
                    )
                    previous_board = None
                time.sleep(0.5)  # Wait for movement to complete
                continue

            consecutive_invalid_moves = 0  # Reset counter on valid move
            previous_board = current_board.copy()
            last_valid_board = current_board.copy()

            fen = board_to_fen(current_board)

            try:
                chess.Board(fen)
                logging.info("FEN is valid")
            except ValueError:
                logging.error("Invalid FEN generated")
                continue

            if fen == previous_fen:
                # The board has not changed, so skip to the next iteration
                time.sleep(1)
                continue
            else:
                save_fen(fen)  # Save the new FEN to the file
                previous_fen = fen

            if playing_color == "w":
                if player_turn:
                    (
                        best_move,
                        best_move_score,
                        random_move_score,
                    ) = get_best_move_with_dynamic_time(
                        engine, fen, piece_count, use_randomizer, current_skill
                    )
                    if best_move is not None:
                        human_readable_move = move_to_readable(
                            best_move, chess.Board(fen)
                        )
                        print(human_readable_move)
                        logging.info(f"Best move: {best_move}")
                        logging.info(human_readable_move)
                        process = psutil.Process(os.getpid())
                        logging.info(f"CPU usage: {psutil.cpu_percent()}%")
                        logging.info(
                            f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB"
                        )

                        # Adjust skill level based on the move quality
                        move_score_difference = abs(best_move_score - random_move_score)
                        current_skill = adjust_skill_level(
                            engine, current_skill, piece_count, move_score_difference
                        )
                    else:
                        logging.error(
                            "Failed to get best move after multiple attempts."
                        )
                    player_turn = False  # It's now the opponent's turn
                else:
                    if board.fen() != fen:
                        board = chess.Board(fen)  # Update the board state
                        player_turn = True  # It's now the player's turn
            else:  # Black player logic
                if player_turn:  # If it's the player's turn, make a move
                    (
                        best_move,
                        best_move_score,
                        random_move_score,
                    ) = get_best_move_with_dynamic_time(
                        engine, fen, piece_count, use_randomizer, current_skill
                    )
                    if best_move is not None:
                        human_readable_move = move_to_readable(
                            best_move, chess.Board(fen)
                        )
                        print(human_readable_move)
                        logging.info(f"Best move: {best_move}")
                        logging.info(human_readable_move)
                        process = psutil.Process(os.getpid())
                        logging.info(f"CPU usage: {psutil.cpu_percent()}%")
                        logging.info(
                            f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB"
                        )

                        # Adjust skill level based on the move quality
                        move_score_difference = abs(best_move_score - random_move_score)
                        current_skill = adjust_skill_level(
                            engine, current_skill, piece_count, move_score_difference
                        )
                    else:
                        logging.error(
                            "Failed to get best move after multiple attempts."
                        )
                    player_turn = False  # It's now the opponent's turn
                else:
                    if board.fen() != fen:
                        board = chess.Board(fen)  # Update the board state
                        player_turn = True  # It's now the player's turn

            if check_win(board_image):
                print("Congratulations! You've won the game!")
                logging.info("Congratulations! You've won the game!")
                action_queue.put("won")
                stop_flag.set()
                break

            time.sleep(1)  # Reduce the sleep time to make it more responsive

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")
            break

    if not stop_flag.is_set():
        action_queue.put("finished")


def handle_user_input():
    while True:
        action = action_queue.get()
        if action == "won":
            while True:
                play_again = (
                    input("Do you want to play another game? (y/n): ").strip().lower()
                )
                if play_again == "y":
                    stop_flag.clear()
                    monitor_thread = threading.Thread(target=monitor_chessboard)
                    monitor_thread.start()
                    break
                elif play_again == "n":
                    print("Thank you for playing! Exiting...")
                    os._exit(0)
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
        elif action == "finished":
            break


monitor_thread = threading.Thread(target=monitor_chessboard)
monitor_thread.start()

user_input_thread = threading.Thread(target=handle_user_input)
user_input_thread.start()

monitor_thread.join()
user_input_thread.join()

# Close the Stockfish engine properly
if "engine" in locals() and engine is not None:
    engine.quit()
