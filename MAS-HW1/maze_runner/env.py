import numpy as np
from typing import Dict, Tuple, List
from enum import IntEnum


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1


class States(IntEnum):
    SL = 0 # Left most 
    SL1 = 1 # Right of SL
    G = 2 # Goal
    SR1 = 3 # Left of SR
    SR = 4 # Right most


class Obs(IntEnum):
    O_3 = 0 # Number of walls around the agent (3 walls)
    O_2 = 1 # Number of walls around the agent (2 walls)


class MazeEnv(object):
    def __init__(self, max_num_steps: int = 1):
        """
        Constructor

        Parameters
        ----------
        max_num_steps
            maximum number of steps allowed in the env
        """
        self.max_num_steps = max_num_steps
        self.num_steps = None
        self.__state = None
        self.done = True

        # define state mapping
        self.__num_states = 3
        self.__state_mapping = {
            States.SL: "SL",
            States.SL1: "SL1",
            States.G: "G",
            States.SR1: "SR1",
            States.SR: "SR"
        }

        # define action mapping
        self.__num_actions = 2
        self.__action_mapping = {
            Actions.LEFT: "Left",
            Actions.RIGHT: "Right",
        }

        # define observation mapping
        self.__num_obs = 2
        self.__obs_mapping = {
            Obs.O_3: "O_3",
            Obs.O_2: "O_2",
        }

        # init transitions & observations probabilities
        # and rewards
        self.__init_transitions()
        self.__init_observations()
        self.__init_rewards()

    def __init_transitions(self):
        # define transition probability for left action
        #       SL  SL1  G  SR1  SR
        # SL	1	0	0	0	0
        # SL1	1	0	0	0	0
        # G	    0	1	0	0	0
        # SR1	0	0	1	0	0
        # SR	0	0	0	1	0
        T_left = np.array([[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ]])

        # define transition probability for the right action
        #       SL  SL1  G  SR1  SR
        # SL	0	1	0	0	0
        # SL1	0	0	1	0	0
        # G	    0	0	0	1	0
        # SR1	0	0	0	0	1
        # SR	0	0	0	0	1
        T_right = np.array([[
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1]
        ]])
        self.__T = np.concatenate([T_left, T_right], axis=0)

    def __init_observations(self):
        # define observation probability for the left action
        #       O_3 O_2
        # SL	1	0
        # SL1	1	0
        # G	    0	1
        # SR1	0	1
        # SR	0	1
        O_left = np.array([[
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1]
        ]])

        # define observation probability for the right action
        #       O_3 O_2
        # SL	0	1
        # SL1	0	1
        # G	    0	1
        # SR1	1	0
        # SR	1	0
        O_right = np.array([[
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0]
        ]])

        self.__O = np.concatenate([O_left, O_right], axis=0)

    def __init_rewards(self):
        # define rewards for left action
        # SL	-1
        # SL1	-1
        # G	    -1
        # SR1	0
        # SR	-1
        R_left = np.array([[-1, -1, -1, 0, -1]])

        # define rewards for right action
        # SL	-1
        # SL1	0
        # G	    -1
        # SR1	-1
        # SR	-1
        R_right = np.array([[-1, 0, -1, -1, -1]])

        self.__R = np.concatenate([R_left, R_right], axis=0)

    def reset(self):
        self.done = False
        self.num_steps = 0

        # initialize the state random
        # this puts the agent int the left and right
        # cell with equal probability
        self.__state = np.random.choice([States.SL, States.SR])

    def step(self, action: Actions) -> Tuple[int, float, bool, Dict[str, int]]:
        """
        Performs an environment step

        Parameters
        ----------
        action
            action to be applied

        Returns
        -------
        Tuple containing the next observation, the reward,
        ending episode flag, other information.
        """
        assert not self.done, "The episode finished. Call reset()!"
        self.num_steps += 1
        self.done = (self.num_steps == self.max_num_steps)

        # get the next observation. this is deterministic
        obs = np.random.choice(
            a=[Obs.O_3, Obs.O_2],
            p=self.O[action][self.__state]
        )

        # get the reward. this is deterministic
        reward = self.R[action][self.__state]

        # get the next transition. this is deterministic
        self.__state = np.random.choice(
            a=[States.SL, States.SL1, States.G, States.SR1, States.SR],
            p=self.T[action][self.__state]
        )

        # construct info
        info = {"num_steps": self.num_steps}
        return obs, reward, self.done, info

    @property
    def state_mapping(self) -> Dict[States, str]:
        """
        Returns
        -------
        State mapping (for display purpose)
        """
        return self.__state_mapping

    @property
    def action_mapping(self) -> Dict[Actions, str]:
        """
        Returns
        -------
        Action mapping (for display purposes)
        """
        return self.__action_mapping

    @property
    def obs_mapping(self) -> Dict[Obs, str]:
        """
        Returns
        -------
        Observation mapping (for display purposes)
        """
        return self.__obs_mapping

    @property
    def T(self) -> np.ndarray:
        """
        Returns
        -------
        Transition probability matrix.
        Axes: (a, s, s')
        """
        return self.__T

    @property
    def O(self) -> np.ndarray:
        """
        Returns
        -------
        Observation probability matrix.
        Axes: (a, s, o)
        """
        return self.__O

    @property
    def R(self) -> np.ndarray:
        """
        Returns
        -------
        Reward matrix:
        Axes: (a, s)
        """
        return self.__R

    @property
    def states(self) -> List[int]:
        """
        Returns
        -------
        List containing the states
        """
        return list(self.__state_mapping.keys())

    @property
    def actions(self) -> List[int]:
        """
        Returns
        -------
        List containing the actions
        """
        return list(self.__action_mapping.keys())

    @property
    def obs(self) -> List[int]:
        """
        Returns
        -------
        List containing the observations
        """
        return list(self.__obs_mapping.keys())