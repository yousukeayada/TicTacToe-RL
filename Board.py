import logging
from enum import IntEnum, auto

import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

class Piece(IntEnum):
    EMPTY = auto()
    BLACK = auto()
    WHITE = auto()

class Winner(IntEnum):
    DRAW  = auto()
    BLACK = auto()
    WHITE = auto()


class Board:
    def __init__(self, size=3):
        self.size = size
        self.reset_stage()

    def put_piece(self, x: int, y: int, piece: Piece) -> None:
        logger.debug(f"Put {piece.name} on ({x}, {y})")
        if piece == Piece.EMPTY:
            raise Exception("Invalid Piece")
        if self.stage[y][x] != Piece.EMPTY:
            raise Exception("Already exists Piece on that position")

        self.stage[y][x] = piece
        self.empties.remove((x, y))

        winner = self.judge(x, y, piece)
        return winner
    
    def can_put(self, x: int, y: int) -> bool:
        return (x, y) in self.empties
    
    def judge(self, x: int, y: int, piece: Piece):
        winner = None
            
        # 行チェック
        cnt = 0
        for i in range(self.size):
            if self.stage[i][x] == piece:
                cnt += 1
        if cnt == self.size:
            if piece == Piece.BLACK:
                winner = Winner.BLACK
            elif piece == Piece.WHITE:
                winner = Winner.WHITE
            return winner
        
        # 列チェック
        cnt = 0
        for j in range(self.size):
            if self.stage[y][j] == piece:
                cnt += 1
        if cnt == self.size:
            if piece == Piece.BLACK:
                winner = Winner.BLACK
            elif piece == Piece.WHITE:
                winner = Winner.WHITE
            return winner

        # 対角線チェック
        cnt = 0
        for i in range(self.size):
            if self.stage[i][i] == piece:
                cnt += 1
        if cnt == self.size:
            if piece == Piece.BLACK:
                winner = Winner.BLACK
            elif piece == Piece.WHITE:
                winner = Winner.WHITE
            return winner

        cnt = 0
        for i in range(self.size):
            if self.stage[i][self.size-i-1] == piece:
                cnt += 1
        if cnt == self.size:
            if piece == Piece.BLACK:
                winner = Winner.BLACK
            elif piece == Piece.WHITE:
                winner = Winner.WHITE
            return winner

        # 揃ってないかつ置けるところがない
        if len(self.empties) == 0:
            winner = Winner.DRAW

        return winner

    def reset_stage(self):
        self.stage = [[Piece.EMPTY for i in range(self.size)] for j in range(self.size)]
        self.empties = [(i, j) for i in range(self.size) for j in range(self.size)]


    def show_stage(self) -> None:
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(self.size):
            for j in range(self.size):
                if self.stage[i][j] == Piece.BLACK:
                    x1.append(j+0.5)
                    y1.append(i+0.5)
                elif self.stage[i][j] == Piece.WHITE:
                    x2.append(j+0.5)
                    y2.append(i+0.5)
        fig = plt.figure()
        ax = fig.add_subplot(aspect="equal")
        ax.grid()
        plt.xlim([0, self.size])
        plt.ylim([0, self.size])
        plt.xticks(np.arange(0, self.size+1, step=1))
        plt.yticks(np.arange(0, self.size+1, step=1))
        ax.scatter(x1, y1, s=1000, marker="o", facecolor="None", edgecolors="blue")
        ax.scatter(x2, y2, s=1000, marker="x", c="red")
        plt.show()

    def test(self):
        self.put_piece(1,1,Piece.BLACK)
        self.put_piece(2,1,Piece.WHITE)
        self.show_stage()
        self.put_piece(1,0,Piece.BLACK)
        self.put_piece(0,1,Piece.WHITE)
        self.show_stage()
        self.put_piece(1,2,Piece.BLACK)
        self.show_stage()
        self.reset_stage()
        self.show_stage()

