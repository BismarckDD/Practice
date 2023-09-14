import numpy as np
import sys
sys.path.append("..")
from agent.base_agent import BaseAgent
from engine.action import Action
from engine.observation import Observation
from algorithm.mcts import MCTS


class PureMctsAgent(BaseAgent):

    def __init__(self):
        super().__init__(self)
        self.mcts = MCTS()

    def get_action(self, obs: Observation) -> Action:
        board = obs.board
        return None

    def get_action_in_training(self, obs: Observation) -> Action:
        return
