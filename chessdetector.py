import os

import cv2

import numpy as np

import pyscreenshot as ImageGrab

import torch

import torchvision.transforms as transforms

from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image

import chess



# Load pre-trained ResNet50 model

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

model.eval()



# Define image preprocessing

preprocess = transforms.Compose([

    transforms.Resize((256, 256)),  # Resize to a fixed size

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



# Capture the screen region where the chessboard is displayed

# Adjust the bbox coordinates according to your screen setup

bbox = (100, 100, 900, 900)  # Example bounding box

screenshot = ImageGrab.grab(bbox).convert('RGB')  # Convert to RGB

screenshot.save("chessboard.png")



# Load the captured image

image = Image.open("chessboard.png").convert('RGB')  # Ensure image is in RGB format

image = preprocess(image)

image = image.unsqueeze(0)



# Function to detect chessboard and apply perspective transformation

def detect_chessboard(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    

    # Find contours

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    

    # Assume the largest contour is the chessboard

    chessboard_contour = max(contours, key=cv2.contourArea)

    

    # Approximate the contour to get the four corners

    epsilon = 0.1 * cv2.arcLength(chessboard_contour, True)

    approx = cv2.approxPolyDP(chessboard_contour, epsilon, True)

    

    if len(approx) == 4:

        # Order points in clockwise order

        pts = approx.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")

        

        s = pts.sum(axis=1)

        rect[0] = pts[np.argmin(s)]

        rect[2] = pts[np.argmax(s)]

        

        diff = np.diff(pts, axis=1)

        rect[1] = pts[np.argmin(diff)]

        rect[3] = pts[np.argmax(diff)]

        

        # Get the width and height of the board

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        maxWidth = max(int(widthA), int(widthB))

        

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxHeight = max(int(heightA), int(heightB))

        

        # Destination points for the perspective transform

        dst = np.array([

            [0, 0],

            [maxWidth - 1, 0],

            [maxWidth - 1, maxHeight - 1],

            [0, maxHeight - 1]

        ], dtype="float32")

        

        # Apply the perspective transform

        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        

        return warped, maxWidth // 8, maxHeight // 8

    

    else:

        raise ValueError("Chessboard not detected correctly")



# Convert the screenshot to an OpenCV image

board_image, cell_width, cell_height = detect_chessboard("chessboard.png")



# Load chess piece templates

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



# Function to match template and identify piece positions

def match_template(piece_image, board_image, threshold=0.7):

    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)

    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(board_gray, piece_gray, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    positions = list(zip(*loc[::-1]))

    return positions



# Initialize an empty board

board = np.full((8, 8), '', dtype=str)



# Identify pieces on the board and draw bounding boxes

for piece, template in piece_templates.items():

    if template is not None:

        positions = match_template(template, board_image, threshold=0.7)

        for pos in positions:

            x, y = pos

            row = y // cell_height

            col = x // cell_width

            board[row, col] = piece[1]

            cv2.rectangle(board_image, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)

            cv2.putText(board_image, piece[1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# Save the annotated board image

cv2.imwrite("annotated_chessboard.png", board_image)



# Convert board matrix to FEN

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

    return fen[:-1]  # Remove the last '/'



fen = board_to_fen(board)

print("FEN:", fen)



# Validate FEN

try:

    chess.Board(fen)

    print("FEN is valid")

except ValueError as e:

    print("FEN is invalid:", e)

