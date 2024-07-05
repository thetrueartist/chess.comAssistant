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



# Configure logging

logging.basicConfig(filename='chess_engine_error.log', level=logging.DEBUG, 

                    format='%(asctime)s %(levelname)s %(message)s')



# Load pre-trained ResNet50 model

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

model.eval()



# Define image preprocessing

preprocess = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



def get_user_input():

    while True:

        playing_color = input("Are you playing as white or black? (w/b): ").strip().lower()

        if playing_color in ('w', 'b'):

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

    return playing_color, skill_level



playing_color, skill_level = get_user_input()



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

    stockfish_path = "/usr/games/stockfish"  # Update this path to where Stockfish is located

    if not os.path.exists(stockfish_path):

        raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    engine.configure({"Threads": threads, "Hash": hash_size})

    return engine



def move_to_readable(move, board):

    piece = board.piece_at(move.from_square)

    piece_name = piece.symbol().upper() if piece else ''

    from_square = chess.SQUARE_NAMES[move.from_square]

    to_square = chess.SQUARE_NAMES[move.to_square]

    piece_map = {

        'P': 'pawn', 'N': 'knight', 'B': 'bishop',

        'R': 'rook', 'Q': 'queen', 'K': 'king'

    }

    return f"\033[1m\033[92mMove {piece_map.get(piece_name, '')} from {from_square} to {to_square}\033[0m"



def capture_screenshot(bbox):

    screenshot_path = "/tmp/chessboard.png"

    screenshot = ImageGrab.grab(bbox).convert('RGB')

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

            dst = np.array([

                [0, 0],

                [maxWidth - 1, 0],

                [maxWidth - 1, maxHeight - 1],

                [0, maxHeight - 1]

            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)

            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            return warped, maxWidth // 8, maxHeight // 8

    return None, None, None



def match_template(piece_image, board_image, threshold=0.7):

    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)

    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(board_gray, piece_gray, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    positions = list(zip(*loc[::-1]))

    return positions



def extract_board_from_image(board_image, piece_templates, cell_width, cell_height, playing_color):

    board = np.full((8, 8), '', dtype=str)

    piece_count = 0  # Initialize piece count

    for piece, template in piece_templates.items():

        if template is not None:

            positions = match_template(template, board_image, threshold=0.7)

            for pos in positions:

                x, y = pos

                row = y // cell_height

                col = x // cell_width

                piece_type = piece[1] if piece[0] == 'b' else piece[1].upper()

                if playing_color == 'b':

                    row, col = 7 - row, 7 - col  # Flip the board for black

                if board[row, col] == '':

                    piece_count += 1

                board[row, col] = piece_type

                cv2.rectangle(board_image, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)

                cv2.putText(board_image, piece_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return board, board_image, piece_count  # Return piece count as well



def board_to_fen(board):

    fen = ""

    for row in board:

        empty_count = 0

        for cell in row:

            if cell == '':

                empty_count += 1

            else:

                if empty_count > 0:

                    fen += str(empty_count)

                    empty_count = 0

                fen += cell

        if empty_count > 0:

            fen += str(empty_count)

        fen += '/'

    fen = fen[:-1]

    if playing_color == 'b':

        fen += " b KQkq - 0 1"

    else:

        fen += " w KQkq - 0 1"

    return fen



def get_dynamic_move_time(piece_count):

    """

    Determines the dynamic move time based on the stage of the game.

    """

    game_stage = get_game_stage(piece_count)

    print(f"Game stage detected: {game_stage}")  # Debug print

    logging.info(f"Game stage detected: {game_stage}")

    if game_stage == 'opening':

        return 1.5  # 1.5 seconds in the opening

    elif game_stage == 'middlegame':

        return 2.5  # 2.5 seconds in the middlegame

    else:  # endgame

        return 4  # 4 seconds in the endgame



def get_game_stage(piece_count):

    """

    Determines the stage of the game: opening, middlegame, or endgame.

    """

    logging.debug(f"Piece count: {piece_count}")

    print(f"Debug - Piece count: {piece_count}")  # Debug print

    

    # Adjust thresholds as needed based on your understanding of the game phases

    if piece_count > 25:

        return 'opening'

    elif 14 < piece_count <= 25:

        return 'middlegame'

    else:

        return 'endgame'



def get_best_move_with_dynamic_time(engine, fen, piece_count, retries=3):

    """

    Attempts to get the best move from Stockfish with dynamic thinking time.

    """

    for attempt in range(retries):

        try:

            board = chess.Board(fen)

            move_time = get_dynamic_move_time(piece_count)

            print(f"Thinking time for this move: {move_time} seconds")  # Debug print

            start_time = time.time()

            result = engine.play(board, chess.engine.Limit(time=move_time))

            end_time = time.time()

            actual_thinking_time = end_time - start_time

            print(f"Actual thinking time: {actual_thinking_time:.2f} seconds")  # Debug print

            logging.info(f"Actual thinking time: {actual_thinking_time:.2f} seconds")

            return result.move

        except Exception as e:

            logging.error(f"Attempt {attempt + 1} - An error occurred with the Stockfish engine: {e}")

            engine = initialize_stockfish()

    return None



def are_images_similar(image1_path, image2_path, threshold=0.1):

    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)

    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:

        return False

    diff = cv2.absdiff(image1, image2)

    non_zero_count = np.count_nonzero(diff)

    total_count = diff.size

    similarity = non_zero_count / total_count

    return similarity < threshold



def monitor_chessboard():

    engine = initialize_stockfish()

    piece_templates = {

        'br': cv2.imread('black_rook.png'),

        'bn': cv2.imread('black_knight.png'),

        'bb': cv2.imread('black_bishop.png'),

        'bq': cv2.imread('black_queen.png'),

        'bk': cv2.imread('black_king.png'),

        'bp': cv2.imread('black_pawn.png'),

        'wr1': cv2.imread('white_rook.png'),

        'wr2': cv2.imread('white_rook_withwhitebackground.png'),

        'wn1': cv2.imread('white_knight.png'),

        'wn2': cv2.imread('white_knight_withgreenbackground.png'),

        'wb1': cv2.imread('white_bishop.png'),

        'wb2': cv2.imread('white_bishop_withgreenbackground.png'),

        'wq1': cv2.imread('white_queen.png'),

        'wq2': cv2.imread('white_queen_withgreenbackground.png'),

        'wk1': cv2.imread('white_king.png'),

        'wk2': cv2.imread('white_king_withwhitebackground.png'),

        'wp1': cv2.imread('white_pawn.png'),

        'wp2': cv2.imread('white_pawn_withgreenbackground.png'),

    }



    board = chess.Board()  # Initialize the chess board

    player_turn = playing_color == 'w'  # True if it's the player's turn

    previous_fen = None

    previous_screenshot_path = None



    while not stop_flag.is_set():

        try:

            screenshot_path = capture_screenshot(bbox)

            if not os.path.exists(screenshot_path):

                logging.error(f"Screenshot file not found: {screenshot_path}")

                continue



            # Compare the current screenshot with the previous one

            if previous_screenshot_path:

                if not are_images_similar(previous_screenshot_path, screenshot_path):

                    previous_screenshot_path = screenshot_path

                    time.sleep(0.5)  # Wait for the animation to complete

                    continue  # Skip this iteration and take a new screenshot



            previous_screenshot_path = screenshot_path



            board_image, cell_width, cell_height = detect_chessboard(screenshot_path)

            if board_image is None:

                print("Chessboard not detected, retrying in 5 seconds...")

                logging.error("Chessboard not detected, retrying in 5 seconds...")

                time.sleep(5)

                continue



            current_board, annotated_board_image, piece_count = extract_board_from_image(board_image, piece_templates, cell_width, cell_height, playing_color)

            cv2.imwrite("annotated_chessboard.png", annotated_board_image)

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

                previous_fen = fen



            if player_turn:

                best_move = get_best_move_with_dynamic_time(engine, fen, piece_count)

                if best_move is not None:

                    human_readable_move = move_to_readable(best_move, chess.Board(fen))

                    print(human_readable_move)

                    logging.info(f"Best move: {best_move}")

                    logging.info(human_readable_move)

                    process = psutil.Process(os.getpid())

                    logging.info(f"CPU usage: {psutil.cpu_percent()}%")

                    logging.info(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")

                else:

                    logging.error("Failed to get best move after multiple attempts.")

                player_turn = False  # It's now the opponent's turn

            else:

                if board.fen() != fen:

                    board = chess.Board(fen)  # Update the board state

                    player_turn = True  # It's now the player's turn



            crown_img = cv2.imread('crown.png', cv2.IMREAD_UNCHANGED)

            if crown_img is not None:

                crown_gray = cv2.cvtColor(crown_img, cv2.COLOR_BGR2GRAY)

                board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

                res = cv2.matchTemplate(board_gray, crown_gray, cv2.TM_CCOEFF_NORMED)

                threshold = 0.8

                loc = np.where(res >= threshold)

                if len(loc[0]) > 0:

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

                play_again = input("Do you want to play another game? (y/n): ").strip().lower()

                if play_again == 'y':

                    stop_flag.clear()

                    monitor_thread = threading.Thread(target=monitor_chessboard)

                    monitor_thread.start()

                    break

                elif play_again == 'n':

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

if 'engine' in locals() and engine is not None:

    engine.quit()

