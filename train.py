import argparse
import sys
import logging
import random

import numpy as np
import matplotlib.pyplot as plt

from TicTacToe import *
from QLAgent import *


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="RL parameter")
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--size', type=int, default=3)
args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
size  = args.size


env = TicTacToe(size=size)
# env.board.show_stage()

num_states = len(Piece) ** env.num_squares
actions    = [i for i in range(env.num_squares)]
agent1 = QLAgent(num_states, actions, alpha=alpha, gamma=gamma) # training
agent2 = QLAgent(num_states, actions) # random

# プロット用
win_cnt      = [dict.fromkeys([w.name for w in Winner], 0) for i in Turn]
win_rate     = [[] for i in Turn]
win_rate_all = []

episode_interval = 1000
for episode in range(100_0000):
    logger.debug(f"--- Start episode {episode+1} ---")
    
    state = env.reset()
    TURN = random.choice(list(Turn))
    turn_count = TURN.value
    logger.debug(f"QLAgent is {TURN.name}")

    while True:
        if turn_count % 2 == Turn.FIRST:
            logger.debug("Agent1's turn")
            while True:
                action = agent1.decide_action(state)
                # action = agent1.decide_random_action()
                logger.debug(f"state: {state}, action: {action}")
                if not env.check(action):
                    logger.debug("Invalid action")
                    agent1.set_q_value(state, action, np.nan)
                    continue
                
                next_state, reward, done, winner = env.step(action, Piece.BLACK)
                logger.debug(f"next state: {next_state}, reward: {reward}, done: {done}")
                if np.isnan(reward):
                    agent1.set_q_value(state, action, reward)
                    continue
                else: # 正しく置けたとき
                    break
            if done:
                logger.debug(f"state: {prev_state} -> {state} -> {next_state}, action: {prev_action} -> {action}, reward: {prev_reward} -> {reward}")
                logger.debug(f"Updates Agent1 (reward: {reward})")
                exp = (state, action, next_state, reward)
                agent1.update_q_table(exp)
                break
        else:
            logger.debug("Agent2's turn")
            while True:
                action = agent2.decide_random_action()
                logger.debug(f"state: {state}, action: {action}")
                if not env.check(action):
                    logger.debug("Invalid action")
                    agent2.q_table[state][action] = np.nan
                    continue
                
                next_state, reward, done, winner = env.step(action, Piece.WHITE)
                logger.debug(f"next state: {next_state}, reward: {reward}, done: {done}")
                if np.isnan(reward):
                    agent2.q_table[state][action] = reward
                    continue
                else: # 正しく置けたとき
                    if reward == 1:
                        reward = -1
                    break
            if done:
                logger.debug(f"state: {prev_state} -> {state} -> {next_state}, action: {prev_action} -> {action}, reward: {prev_reward} -> {reward}")
                logger.debug(f"Updates Agent1 (reward: {reward})")
                exp = (prev_state, prev_action, next_state, reward)
                agent1.update_q_table(exp)
                break

        prev_state, prev_action, prev_reward = state, action, reward
        state = next_state
        turn_count += 1
        # env.board.show_stage()


    logger.debug(f"Result: {winner.name}")
    win_cnt[TURN.value][winner.name] += 1
    win_rate[TURN.value].append(win_cnt[TURN.value][Winner.BLACK.name] / sum(win_cnt[TURN.value].values()))
    win_rate_all.append((win_cnt[0][Winner.BLACK.name]+win_cnt[1][Winner.BLACK.name]) / (episode+1))
    logger.debug(f"--- Finish episode {episode+1} ---")
    if episode % episode_interval == 0:
        logger.info(f"--- Finish episode {episode+1} ---")
    # env.board.show_stage()


logger.info(win_cnt)
logger.info(f"WP: {win_rate[0][-1]} {win_rate[1][-1]} {win_rate_all[-1]}")

logger.info(f"alpha: {alpha}, gamma: {gamma}")
print(f"{win_rate[0][-1]} \n{win_rate[1][-1]} \n{win_rate_all[-1]}")
for cnt in win_cnt:
    for k, v in cnt.items():
        print(v)

plt.plot(win_rate_all, label="Total")
plt.plot(win_rate[0], label="First")
plt.plot(win_rate[1], label="Second")
plt.ylim([0,1])
plt.xlabel("Episode")
plt.ylabel("Winning percentage")
plt.legend()
plt.savefig("wp.png")

agent1.save_q_table(f"q_table_{size}")
