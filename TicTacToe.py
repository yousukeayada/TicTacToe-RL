from itertools import product
from enum import IntEnum, auto

import numpy as np

from Board import *


class Turn(IntEnum):
    FIRST  = 0
    SECOND = 1


class TicTacToe:
    def __init__(self, size=3):
        self.size        = size
        self.num_squares = self.size * self.size

        self.board   = Board(size=self.size)

    def reset(self):
        self.board.reset_stage()
        state = 0
        return state

    def step(self, action, piece):
        x, y = action % self.size, int(action / self.size)
        try:
            winner = self.board.put_piece(x, y, piece)

            next_state = self.convert_to_state(self.board.stage)

            if winner:
                done = True
                if winner == Winner.DRAW:
                    reward = 0
                else:
                    reward = 1
            else:
                reward, done = 0, False
            return next_state, reward, done, winner
        except Exception as e:
            logger.info(e)
            return None, np.nan, False, None

    def check(self, action):
        x, y = action % self.size, int(action / self.size)
        return self.board.can_put(x, y)
        
    def convert_to_state(self, stage):
        s = [stage[i][j] for i in range(self.size) for j in range(self.size)]
        index = 0
        for i in range(self.num_squares):
            index += (s[i]-1) * (len(Piece) ** (self.num_squares-i-1))
        return index
