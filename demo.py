import argparse
import logging

import numpy as np

from TicTacToe import *
from QLAgent import *


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="RL parameter")
parser.add_argument('--size', type=int, default=3)
args = parser.parse_args()

size  = args.size


env = TicTacToe(size=size)

num_states = len(Piece) ** env.num_squares
actions    = [i for i in range(env.num_squares)]
agent = QLAgent(num_states, actions)

state = env.reset()
agent.load_q_table(f"q_table_demo_{size}.npy")

env.board.show_stage()

# 先攻後攻を決める
while True:
    try:
        logger.info("Input 0 (first) or 1 (second)")
        turn_count = Turn(int(input()))
        logger.info(f"You are {turn_count.name}")
        break
    except Exception as e:
        logger.info(e)
        continue
    if turn_count != Turn.FIRST and turn_count != Turn.SECOND:
        logger.info("Invalid input")
        continue


while True:
    if turn_count % 2 == Turn.FIRST:
        logger.info("Player's turn")
        while True:
            try:
                logger.info("Input number 0~8")
                action = int(input())
            except Exception as e:
                logger.info(e)
                continue
            logger.debug(f"state: {state}, action: {action}")
            if action < 0 or action >= 9:
                logger.info("Invalid input")
                continue
            if not env.check(action):
                logger.info("Invalid action")
                continue
            
            next_state, reward, done, winner = env.step(action, Piece.WHITE)
            logger.debug(f"next state: {next_state}, reward: {reward}, done: {done}")
            if np.isnan(reward):
                continue
            else: # 正しく置けたとき
                break
        if done:
            logger.debug(f"state: {prev_state} -> {state} -> {next_state}, action: {prev_action} -> {action}, reward: {prev_reward} -> {reward}")
            break

    elif (turn_count % 2) == Turn.SECOND:
        logger.info("Agent's turn")
        while True:
            action = agent.decide_optimal_action(state)
            logger.debug(f"state: {state}, action: {action}")
            if not env.check(action):
                logger.info("Invalid action")
                agent.set_q_value(state, action, np.nan)
                continue
            
            next_state, reward, done, winner = env.step(action, Piece.BLACK)
            logger.debug(f"next state: {next_state}, reward: {reward}, done: {done}")
            if np.isnan(reward):
                agent.set_q_value(state, action, reward)
                continue
            else: # 正しく置けたとき
                break
        if done:
            logger.debug(f"state: {prev_state} -> {state} -> {next_state}, action: {prev_action} -> {action}, reward: {prev_reward} -> {reward}")
            break

    prev_state, prev_action, prev_reward = state, action, reward
    state = next_state

    env.board.show_stage()
    turn_count += 1


logger.info(f"Result: {winner.name}")
env.board.show_stage()
