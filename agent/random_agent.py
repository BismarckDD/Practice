import random

from engine.action import Action
from engine.observation import Observation
from agent.base_agent import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        return

    def get_action(self, obs: Observation) -> (Action, {}):
        try:
            action = obs.available_actions[random.randint(0, len(obs.available_actions) - 1)]
        except Exception as e:
            print("reason:", len(obs.available_actions), str(e))
            action = None
        finally:
            if action is None:
                exit(-1)
            else:
                return action, {}
