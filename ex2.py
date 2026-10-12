import math

# Create the board
board = [' ' for _ in range(9)]

# Display the board
def print_board():
    print()
    print(board[0] + " | " + board[1] + " | " + board[2])     # first row
    print("--+---+--")
    print(board[3] + " | " + board[4] + " | " + board[5])     # second row
    print("--+---+--")
    print(board[6] + " | " + board[7] + " | " + board[8])     # last row
    print()

# Check winner
def check_winner(player):
    win_positions = [                 # stores all possible winning combinations
        [0,1,2], [3,4,5], [6,7,8],   # Rows
        [0,3,6], [1,4,7], [2,5,8],   # Columns
        [0,4,8], [2,4,6]             # Diagonals
    ]

    for pos in win_positions:             # Checks every winning combination one by one
        if board[pos[0]] == board[pos[1]] == board[pos[2]] == player:     # checks whether all three positions contain the same player's symbol
            return True                   # if yes, player has won
    return False                          # if no winning combination is found

# Check draw
def is_draw():
    return ' ' not in board             # checks if there are no empty spaces left

# Minimax Algorithm
def minimax(is_maximizing):           # tells whose turn it is - Computer's turn (Maximizer), Player's turn (Minimizer)

    if check_winner('O'):             # if computer wins, Score = +1
        return 1

    if check_winner('X'):             # if player wins, Score = -1
        return -1

    if is_draw():                     # if game is tied, Score = 0
        return 0

    if is_maximizing:                 # if computer's turn, computer tries to maximize the score

        best_score = -math.inf          # initial max score is - infinite

        for i in range(9):              # check every square
            if board[i] == ' ':
                board[i] = 'O'

                score = minimax(False)    # call minimax

                board[i] = ' '            # explore the possible outcomes after this move

                best_score = max(score, best_score)         # choose larger score

        return best_score

    else:                                 # minimizing player (human)

        best_score = math.inf             # initialize +infinite

        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'

                score = minimax(True)         # Now it becomes the computer's turn.

                board[i] = ' '                # remove the temporary move

                best_score = min(score, best_score)     # Choose the smallest score

        return best_score

# Computer chooses the best move
def computer_move():

    best_score = -math.inf
    move = -1

    for i in range(9):

        if board[i] == ' ':

            board[i] = 'O'

            score = minimax(False)

            board[i] = ' '

            if score > best_score:
                best_score = score
                move = i

    board[move] = 'O'

# Main Program
print("TIC-TAC-TOE USING MINIMAX")
print("You are X")
print("Computer is O")

while True:                   # Runs the game until someone wins or the game is a draw.

    print_board()

    player = int(input("Enter position (1-9): ")) - 1

    if board[player] != ' ':                  # Checks whether the position is already occupied
        print("Invalid Move! Try Again.")
        continue

    board[player] = 'X'

    if check_winner('X'):
        print_board()
        print("You Win!")
        break

    if is_draw():
        print_board()
        print("Match Draw!")
        break

    computer_move()

    if check_winner('O'):
        print_board()
        print("Computer Wins!")
        break

    if is_draw():
        print_board()
        print("Match Draw!")
        break