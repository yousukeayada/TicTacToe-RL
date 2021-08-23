import random
import logging

import numpy as np

from Agent import *


logger = logging.getLogger(__name__)

class QLAgent(Agent):
    def __init__(self, num_states, actions, alpha=0.1, gamma=0.9):
        self.num_states  = num_states
        self.actions     = actions
        self.num_actions = len(actions)

        self.rng = np.random.default_rng()
        # self.q_table = self.rng.uniform(-1, 1, size=(self.num_states, self.num_actions))
        self.q_table = np.zeros((self.num_states, self.num_actions))

        self.epsilon = 0.1

        self.alpha = alpha
        self.gamma = gamma

    def decide_action(self, state):
        if self.rng.uniform() < self.epsilon:
            return self.decide_random_action()
        else:
            return self.decide_optimal_action(state)

    def decide_random_action(self):
        return random.choice(self.actions)

    def decide_optimal_action(self, state):
        return np.nanargmax(self.q_table[state])

    def update_q_table(self, exp):
        state, action, next_state, reward = exp
        q_s_a    = self.q_table[state][action]
        max_q_ns = np.nanmax(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_q_ns - q_s_a)
        logger.debug(f"Q({state},{action}): {q_s_a} -> {self.q_table[state][action]}")

    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

    def save_q_table(self, path):
        # np.save(path, self.q_table)
        np.savez_compressed(path, q_table=self.q_table)

    def load_q_table(self, path):
        logger.info("Load Q table ...")
        # self.q_table = np.load(path)
        self.q_table = np.load(path)["q_table"]

    def test(self):
        state = 500
        print("Policy action\tRandom action")
        for i in range(10):
            print(f"{i+1} 回目: {self.decide_action(state)}\t{self.decide_random_action()}")
        print(f"Optimal action: {self.decide_optimal_action(state)}")

        print("Training...")
        action, next_state, reward = 1, 1000, 1
        exp = (state, action, next_state, reward)
        for i in range(10):
            self.update_q_table(exp)
        print(f"Optimal action: {self.decide_optimal_action(state)}")

