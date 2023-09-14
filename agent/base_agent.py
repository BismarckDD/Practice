from engine.action import Action
from engine.observation import Observation


class BaseAgent(object):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.__init__()

    def get_action(self, obs: Observation) -> (Action, {}):
        return None, None

    def get_action_in_training(self, obs: Observation) -> (Action, {}):
        return None, None
