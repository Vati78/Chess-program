import pygame as pg
import time

pg.init()

"""
Constants and initialization
"""
SQUARE = 70
WIDTH = 8*SQUARE
HEIGHT = 8*SQUARE

win = pg.display.set_mode((WIDTH+300, HEIGHT))
pg.display.set_caption("Chess")
win.fill((50, 50, 50))

GREEN = (118,150,86)
WHITE = (238,238,210)
GREY = (50, 50, 50)


# variables
position = [["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]

k_move = {"w": False,
          "b": False}

a_rook_move = {"w":False,
                "b" : False}
h_rook_move = {"w":False,
                "b" : False}

k_pos = {"w": (4, 7),
         "b": (4, 0)}

last_move = (None, None, None, None, None)

liste_position = []

"""
Functions
"""
# draws the board
def draw_board():
    for col in range(8):
        for rank in range(8):
            if (col+rank) % 2 == 0:
                pg.draw.rect(win, WHITE, (col*SQUARE, rank*SQUARE, SQUARE, SQUARE))
            else:
                pg.draw.rect(win, GREEN, (col * SQUARE, rank * SQUARE, SQUARE, SQUARE))

# draws the pieces
def draw_pieces():
    global position
    for rank in enumerate(position):
        for col in enumerate(rank[1]):
            if col[1] != " ":
                win.blit(pg.transform.scale(pg.image.load(f"Pieces/{col[1]}.png"), (SQUARE, SQUARE)), ((col[0])*SQUARE, (rank[0])*SQUARE))

# translates mouse position to chess coordinates
def mouse_to_pos(m_pos):
    m_x, m_y = m_pos
    if 0<=m_x<=WIDTH and 0<=m_y<=HEIGHT:
        return True, (m_x//SQUARE, m_y//SQUARE)
    else:
        return False, None

# checks if there is a piece on a certain square
def if_piece(col, rank):
    global position
    square = position[rank][col]
    if square != " ":
        return square
    else:
        return False

# gets the type of piece on the square
def get_type(col, rank):
    piece = position[rank][col][1]
    if piece == " ":
        return None
    if piece == "R":
        return Piece().Rook()
    elif piece == "P":
        return Piece().Pawn()
    elif piece == "K":
        return Piece().King()
    elif piece == "N":
        return Piece().Knight()
    elif piece == "Q":
        return Piece().Queen()
    elif piece == "B":
        return Piece().Bishop()

# gets the color of the piece*
def get_color(col, rank):
    return position[rank][col][0]

# checks if the move is legal
def is_valid_move(player, col_i, rank_i, col_f, rank_f):
    if (col_f, rank_f) in remove_illegal(player, col_i, rank_i, get_type(col_i, rank_i).legal_moves(col_i, rank_i)):
        return True
    else:
        return False

# plays the move
def move(col_i, col_f, rank_i, rank_f):
    global position

    piece = position[rank_i][col_i]

    position[rank_f][col_f] = piece
    position[rank_i][col_i] = " "

# blits all the legal moves
def blit_legal_moves(liste):
    for move in liste:
        col = move[0]
        rank = move[1]
        if if_piece(col, rank):
            pg.draw.circle(win, (168,168,168), (col*SQUARE+SQUARE//2, rank*SQUARE+SQUARE//2), SQUARE//2-2, 3)
        else:
            pg.draw.circle(win, (168,168,168), (col * SQUARE + SQUARE // 2, rank * SQUARE + SQUARE // 2), 10)

# checks if there is a check
def in_check(player, col_k, rank_k):
    global position

    opposite_color = "b" if player == "w" else "w"

    for rank in enumerate(position):
        for col in enumerate(rank[1]):
            if position[rank[0]][col[0]][0] == opposite_color and position[rank[0]][col[0]][1] != "K":
                moves = get_type(col[0], rank[0]).legal_moves(col[0], rank[0])
                for move in moves:
                    if move == (col_k, rank_k):
                        return True
    return False

# remove illegal moves if check
def remove_illegal(player, col_i, rank_i, liste):
    global position

    legal_moves = []
    if position[rank_i][col_i][1] == "K":
        king = True
    else:
        king = False

    for moves in liste:
        last_piece = position[moves[1]][moves[0]]

        move(col_i, moves[0], rank_i, moves[1])

        if king:
            k_col, k_rank = moves
        else:
            k_col = k_pos[player][0]
            k_rank = k_pos[player][1]


        if not in_check(player, k_col, k_rank):
            legal_moves.append(moves)

        move(moves[0], col_i, moves[1], rank_i)
        position[moves[1]][moves[0]] = last_piece

    return legal_moves

# checks if checkmate
def checkmate(player):
    global position

    if in_check(player, k_pos[player][0], k_pos[player][1]):
        for rank in enumerate(position):
            for col in enumerate(rank[1]):
                if position[rank[0]][col[0]][0] == player:
                    if len(remove_illegal(player, col[0], rank[0], get_type(col[0], rank[0]).legal_moves(col[0], rank[0]))) != 0:
                        return False
        return True

    else:
        return False

# checks if stalemate
def stalemate(player):
    global position

    if not in_check(player, k_pos[player][0], k_pos[player][1]):
        for rank in enumerate(position):
            for col in enumerate(rank[1]):
                if position[rank[0]][col[0]][0] == player:
                    if len(remove_illegal(player, col[0], rank[0],
                                          get_type(col[0], rank[0]).legal_moves(col[0], rank[0]))) != 0:
                        return False
        return True

    else:
        return False

"""
Pieces
"""
class Piece:
    global position

    class Pawn:
        def legal_moves(self, col_i, rank_i):
            color = get_color(col_i, rank_i)

            legal_moves = []

            if color == "w":
                if rank_i == 6:
                    if not if_piece(col_i, rank_i-2) and not if_piece(col_i, rank_i-1):
                        legal_moves.append((col_i, rank_i-2))

                if not if_piece(col_i, rank_i-1):
                    legal_moves.append((col_i, rank_i-1))

                if 1 <= col_i <= 7 and position[rank_i-1][col_i-1][0] == "b":
                    legal_moves.append((col_i-1, rank_i-1))

                if 0 <= col_i <= 6 and position[rank_i-1][col_i+1][0] == "b":
                    legal_moves.append((col_i+1, rank_i-1))

                # en passant
                if (last_move[0] == "bP"
                    and last_move[3] == 1
                    and last_move[4] == 3
                    and rank_i == 3):
                    if col_i == last_move[2]+1:
                        legal_moves.append((col_i-1, 2))
                    elif col_i == last_move[2]-1:
                        legal_moves.append((col_i+1, 2))


            elif color == "b":
                if rank_i == 1:
                    if not if_piece(col_i, rank_i+2):
                        legal_moves.append((col_i, rank_i+2))

                if not if_piece(col_i, rank_i+1):
                    legal_moves.append((col_i, rank_i+1))

                if 1 <= col_i <= 7 and position[rank_i+1][col_i-1][0] == "w":
                    legal_moves.append((col_i-1, rank_i+1))

                if 0 <= col_i <= 6 and position[rank_i+1][col_i+1][0] == "w":
                    legal_moves.append((col_i+1, rank_i+1))

                # en passant
                if (last_move[0] == "wP"
                    and last_move[3] == 6
                    and last_move[4] == 4
                    and rank_i == 4):
                    if col_i == last_move[2]+1:
                        legal_moves.append((col_i-1, 5))
                    elif col_i == last_move[2]-1:
                        legal_moves.append((col_i+1, 5))

            return legal_moves

    class Rook:
        def legal_moves(self, col_i, rank_i):
            color = get_color(col_i, rank_i)

            dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            legal_moves = []

            for direction in dir:
                col = col_i
                rank = rank_i
                while 0 <= col <= 7 and 0 <= rank <= 7:
                    rank += direction[0]
                    col += direction[1]

                    if col < 0 or col > 7 or rank < 0 or rank > 7:
                        break

                    if position[rank][col][0] != color:
                        legal_moves.append((col, rank))
                        if position[rank][col] != " ":
                            break
                    else:
                        break
            return legal_moves

    class Knight:
        def legal_moves(self, col_i, rank_i):
            color = get_color(col_i, rank_i)

            dir = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]

            legal_moves = []

            for direction in dir:
                rank = rank_i+direction[0]
                col = col_i+direction[1]

                if 0 <= rank <= 7 and 0 <= col <= 7:
                    if position[rank][col][0] != color:
                        legal_moves.append((col, rank))

            return legal_moves

    class Bishop:
        def legal_moves(self, col_i, rank_i):
            color = get_color(col_i, rank_i)

            dir = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

            legal_moves = []

            for direction in dir:
                col = col_i
                rank = rank_i
                while 0 <= col <= 7 and 0 <= rank <= 7:
                    rank += direction[0]
                    col += direction[1]

                    if col < 0 or col > 7 or rank < 0 or rank > 7:
                        break

                    if position[rank][col][0] != color:
                        legal_moves.append((col, rank))
                        if position[rank][col] != " ":
                            break
                    else:
                        break

            return legal_moves

    class King:
        def legal_moves(self, col_i, rank_i):
            color = get_color(col_i, rank_i)

            dir = [(1, 1), (1, -1), (1, 0), (-1, 1), (-1, -1), (-1, 0), (0, 1), (0, -1)]

            legal_moves = []

            for direction in dir:
                col = col_i + direction[1]
                rank = rank_i + direction[0]

                if 0 <= col <= 7 and 0 <= rank <= 7:
                    if position[rank][col][0] != color and not in_check(color, col, rank):
                        legal_moves.append((col, rank))


            # short castle
            if (not k_move[color]
                    and not h_rook_move[color]
                    and not in_check(color, 4, rank_i)
                    and not in_check(color, 5, rank_i)
                    and not in_check(color, 6, rank_i)
                    and not if_piece(5, rank_i)
                    and not if_piece(6, rank_i)):
                legal_moves.append((6, rank_i))

            # long castle
            if (not k_move[color]
                    and not a_rook_move[color]
                    and not in_check(color, 2, rank_i)
                    and not in_check(color, 3, rank_i)
                    and not in_check(color, 4, rank_i)
                    and not if_piece(3, rank_i)
                    and not if_piece(2, rank_i)):
                legal_moves.append((2, rank_i))

            return legal_moves

    class Queen:
        def legal_moves(self, col_i, rank_i):
            rook_moves = Piece.Rook().legal_moves(col_i, rank_i)
            bishop_moves = Piece.Bishop().legal_moves(col_i, rank_i)

            legal_moves = []

            for move in rook_moves:
                legal_moves.append(move)
            for move in bishop_moves:
                legal_moves.append(move)

            return legal_moves

"""
Game
"""
def main():
    global k_move, a_rook_move, h_rook_move, k_pos, last_move, liste_position

    running = True

    col_i = None
    col_f = None
    rank_i = None
    rank_f = None

    move_i = False
    move_f = False

    player = "w"

    while running:
        mouse_pos = pg.mouse.get_pos()

        for event in pg.event.get():
            # if the player quits
            if event.type == pg.QUIT:
                running = False

            # if the mouse button is down
            if event.type == pg.MOUSEBUTTONDOWN:
                val, get_pos = mouse_to_pos(mouse_pos)
                if val and position[get_pos[1]][get_pos[0]] != " ":
                    move_i = True
                    col_i = get_pos[0]
                    rank_i = get_pos[1]
                else:
                    move_i = False

            # if the mouse button is up
            if event.type == pg.MOUSEBUTTONUP:
                if 0<=mouse_pos[0]<=WIDTH and 0<=mouse_pos[1]<=HEIGHT:
                    val, get_pos = mouse_to_pos(mouse_pos)
                    if val and move_i:
                        move_f = True
                        col_f = get_pos[0]
                        rank_f = get_pos[1]
                    else:
                        move_f = False
                else:
                    move_f = False
                    move_i = False

        # draws the board and the pieces
        win.fill(GREY)
        draw_board()
        draw_pieces()

        #if the player drags a piece
        if move_i:
            # empty the square where the piece comes from
            if (col_i + rank_i) % 2 == 0:
                pg.draw.rect(win, WHITE, (col_i * SQUARE, rank_i * SQUARE, SQUARE, SQUARE))
            else:
                pg.draw.rect(win, GREEN, (col_i * SQUARE, rank_i * SQUARE, SQUARE, SQUARE))

            # draws all the possible moves
            if position[rank_i][col_i][0] == player:
                blit_legal_moves(remove_illegal(player, col_i, rank_i, get_type(col_i, rank_i).legal_moves(col_i, rank_i)))

            # draws the piece where the mouse is
            win.blit(pg.transform.scale(pg.image.load(f"Pieces/{position[rank_i][col_i]}.png"),
                                        (SQUARE + 20, SQUARE + 20)),
                                        (mouse_pos[0]-(SQUARE+20)//2, mouse_pos[1]-(SQUARE+20)//2))



        # if a move is played
        if move_f and move_i:
            # if it's not the same square
            if (col_i,rank_i) != (col_f,rank_f):
                # if the move is valid and if it's the right player's turn
                if is_valid_move(player, col_i, rank_i, col_f, rank_f) and get_color(col_i, rank_i) == player:
                    piece = position[rank_i][col_i]
                    # if the king is moved
                    if piece[1] == "K":
                        k_move[player] = True
                        k_pos[player] = (col_f, rank_f)

                        # if short castle
                        if col_f == col_i + 2:
                            move(7, 5, rank_i, rank_f)
                        # if long castle
                        elif col_f == col_i - 2:
                            move(0, 3, rank_i, rank_f)

                    # if a rook is moved
                    elif piece[1] == "R":
                        if col_i == 0:
                            a_rook_move[player] = True
                        elif col_i == 7:
                            h_rook_move[player] = True

                    # if a pawn is moved
                    elif piece[1] == "P":
                        # if white promotes
                        if player == "w":
                            # if en passant IS PLAYED (which it is because en passant is always forced :))
                            if (last_move[0] == "bP"
                                    and last_move[3] == 1
                                    and last_move[4] == 3
                                    and rank_i == 3):
                                if col_i == last_move[2] + 1:
                                    position[3][col_i - 1] = " "
                                elif col_i == last_move[2] - 1:
                                    position[3][col_i + 1] = " "

                            # if promotes
                            if rank_f == 0:
                                pg.draw.rect(win, (255, 255, 255), (col_f*SQUARE, 0, SQUARE, 4*SQUARE))

                                list_pieces = ["wQ", "wR", "wN", "wB"]

                                for pieces in enumerate(list_pieces):
                                    win.blit(pg.transform.scale(pg.image.load(f"Pieces/{pieces[1]}.png"), (SQUARE, SQUARE)),
                                             (col_f * SQUARE, pieces[0]*SQUARE))

                                pg.display.update()

                                promote = True

                                while promote:
                                    mouse_pos = pg.mouse.get_pos()

                                    val, sq = mouse_to_pos(mouse_pos)

                                    for event in pg.event.get():
                                        if event.type == pg.MOUSEBUTTONDOWN:
                                            if val and 0 <= sq[1] <= 3 and sq[0] == col_f:
                                                position[1][col_i] = list_pieces[sq[1]]
                                                promote = False

                        else:
                            # if en passant is played
                            if (last_move[0] == "wP"
                                  and last_move[3] == 6
                                  and last_move[4] == 4
                                  and rank_i == 4):
                                if col_i == last_move[2] + 1:
                                    position[4][col_i - 1] = " "
                                elif col_i == last_move[2] - 1:
                                    position[4][col_i + 1] = " "

                            # if promotes
                            if rank_f == 7:
                                pg.draw.rect(win, (255, 255, 255), (col_f*SQUARE, HEIGHT-4*SQUARE, SQUARE, 4*SQUARE))

                                list_pieces = ["bQ", "bR", "bN", "bB"]

                                for pieces in enumerate(list_pieces):
                                    win.blit(pg.transform.scale(pg.image.load(f"Pieces/{pieces[1]}.png"), (SQUARE, SQUARE)),
                                             (col_f*SQUARE, HEIGHT-(pieces[0]+1)*SQUARE))

                                pg.display.update()

                                promote = True

                                while promote:
                                    mouse_pos = pg.mouse.get_pos()

                                    val, sq = mouse_to_pos(mouse_pos)

                                    for event in pg.event.get():
                                        if event.type == pg.MOUSEBUTTONDOWN:
                                            if val and 4 <= sq[1] <= 7 and sq[0] == col_f:
                                                position[6][col_i] = list_pieces[7-sq[1]]
                                                promote = False




                    move(col_i, col_f, rank_i, rank_f)
                    last_move = (piece, col_i, col_f, rank_i, rank_f)

                    # checks if 3-fold repetition
                    if player == "b":
                        liste_position.append([rank[:] for rank in position])
                        if liste_position.count(position) == 3:
                            print("3-fold repetition")

                    player = "w" if player == "b" else "b"

            move_f = False
            move_i = False

        if checkmate(player) or stalemate(player):
            print("checkmate or stalemate")
            #pg.draw.rect(win, GREY, (200, 200, 300, 200), border_radius=10)
            # win.blit(pg.font.Font(None, 36).render("White won !", True, (0, 0, 0)), (220, 210))


        pg.display.update()


if __name__ == "__main__":
    main()