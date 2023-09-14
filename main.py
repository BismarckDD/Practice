# This is a sample Python script.
import random
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
sys.path.append("..")
from engine.game import Game
from agent.random_agent import RandomAgent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = Game()
    agent1 = RandomAgent()
    # agent2 = HumanAgent()
    agent2 = RandomAgent()
    obs = game.reset()
    # print(actions)
    while game.is_terminated()[0] is False:
        if game.get_round() % 2 == 1:
            action, _ = agent1.get_action(obs)
        else:
            action, _ = agent2.get_action(obs)
        obs = game.step(action)
        game.show()
    game.get_observation()
    exit(0)
