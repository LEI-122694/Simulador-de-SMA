# Agent.py
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for all agents.
    Both LighthouseAgent and MazeAgent inherit from this.
    """

    def __init__(self, name, env, start_pos):
        self.name = name
        self.env = env
        self.x, self.y = start_pos

        # Register the agent in the environment
        self.env.add_agent(self)

        # Last received observation
        self.current_obs = None

        self.mode = "train"

    # ------------------------------------------------------------------
    # Required API (forced by assignment but abstracted here)
    # ------------------------------------------------------------------

    @classmethod
    def cria(cls, ficheiro):
        """
        Factory method required by assignment.
        Child classes may override if needed.
        """
        print("[AGENTE] carregar parâmetros (stub)")
        return None

    @abstractmethod
    def observacao(self, obs):
        """
        Store/store-process the observation from the environment.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def age(self):
        """
        Decide the action and return the next position (x,y) or None.
        Must be implemented by subclasses.
        """
        pass

    def avaliacaoEstadoAtual(self, recompensa):
        """Optional for learning; left as stub."""
        pass

    def instala(self, sensor):
        """Optional, not needed now."""
        pass

    # ------------------------------------------------------------------
    # Communication API
    # ------------------------------------------------------------------

    @abstractmethod
    def comunica(self, mensagem, de_agente):
        """
        Handle a message from another agent.
        Child classes decide the content and format.
        """
        pass

    def _broadcast(self, msg):
        """
        Send a message to all other agents in the environment.
        """
        for other in self.env.agents:
            if other is not self:
                other.comunica(msg, self)

    # ------------------------------------------------------------------
    # Optional helper: update agent position (called by env after move)
    # ------------------------------------------------------------------
    def set_position(self, x, y):
        self.x, self.y = x, y

    def set_mode(self, mode):
        assert mode in ("train", "test")
        self.mode = mode