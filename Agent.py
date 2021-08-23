from abc import *

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def decide_action(self):
        raise NotImplementedError()

        