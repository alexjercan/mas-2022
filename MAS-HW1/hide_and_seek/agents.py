import numpy as np

from hide_and_seek.envs import HideAndSeekEnv, GameState, GameAction, Action
from hide_and_seek.envs import GridPosition
from random import choice
from collections import defaultdict
from itertools import product

PRODUCT_ACTIONS = list(product(Action, repeat=2))


class ConstScheduler:
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def __call__(self):
        return self.epsilon

    def __str__(self):
        return f"ConstScheduler(epsilon={self.epsilon})"


class LinearScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.001):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def __call__(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return self.epsilon

    def __str__(self):
        return f"LinearScheduler(epsilon_start={self.epsilon_start}, epsilon_end={self.epsilon_end}, epsilon_decay={self.epsilon_decay})"


class ExponentialScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def __call__(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def __str__(self):
        return f"ExponentialScheduler(epsilon_start={self.epsilon_start}, epsilon_end={self.epsilon_end}, epsilon_decay={self.epsilon_decay})"


class SimpleAgent:
    ag_ids = [1, 2]

    def __init__(self):
        self.total_reward = 0

    def get_action(self, state: GameState):
        return {ag_id: choice([e for e in Action]) for ag_id in SimpleAgent.ag_ids}


def epsilon_greedy(Q: dict, state: GameState, epsilon=0.1, train=True):
    if np.random.random() < epsilon and train:
        return np.random.randint(0, Q[state].shape[0])
    else:
        return np.argmax(Q[state])


class QLearningHider(SimpleAgent):
    def __init__(
        self,
        env: HideAndSeekEnv,
        seeker: SimpleAgent,
        num_episodes=100,
        discount_factor=1.0,
        alpha=0.5,
        epsilon_scheduler=ConstScheduler(0.1),
        max_iter=100_000,
    ):
        super(QLearningHider, self).__init__()

        stats = np.zeros(num_episodes)
        Q = defaultdict(lambda: np.zeros(env.get_num_actions()))

        for i_episode in range(num_episodes):
            state = env.reset()

            for i in range(max_iter):
                action_index = epsilon_greedy(Q, state, epsilon_scheduler())
                action = {
                    ag_id: PRODUCT_ACTIONS[action_index][ag_id - 1]
                    for ag_id in SimpleAgent.ag_ids
                }
                action = GameAction(
                    seeker_actions=seeker.get_action(state),
                    hider_actions=action,
                )

                next_state, reward, done = env.step(action)

                stats[i_episode] += reward.total_hider_reward

                Q[state][action_index] += alpha * (
                    reward.total_hider_reward
                    + discount_factor * np.max(Q[next_state])
                    - Q[state][action_index]
                )

                if done:
                    break

                state = next_state

        self.Q = Q
        self.stats = stats

    def get_action(self, state: GameState):
        action_index = epsilon_greedy(self.Q, state, train=False)
        action = {
            ag_id: PRODUCT_ACTIONS[action_index][ag_id - 1]
            for ag_id in SimpleAgent.ag_ids
        }

        return action


class SarsaHider(SimpleAgent):
    def __init__(
        self,
        env: HideAndSeekEnv,
        seeker: SimpleAgent,
        num_episodes=100,
        discount_factor=1.0,
        alpha=0.5,
        epsilon_scheduler=ConstScheduler(0.1),
        max_iter=100_000,
    ):
        super(SarsaHider, self).__init__()

        stats = np.zeros(num_episodes)
        Q = defaultdict(lambda: np.zeros(env.get_num_actions()))

        for i_episode in range(num_episodes):
            state = env.reset()

            action_index = epsilon_greedy(Q, state, epsilon_scheduler())

            for i in range(max_iter):
                action = {
                    ag_id: PRODUCT_ACTIONS[action_index][ag_id - 1]
                    for ag_id in SimpleAgent.ag_ids
                }
                action = GameAction(
                    seeker_actions=seeker.get_action(state),
                    hider_actions=action,
                )
                next_state, reward, done = env.step(action)

                stats[i_episode] += reward.total_hider_reward

                next_action_index = epsilon_greedy(Q, next_state, epsilon_scheduler())

                Q[state][action_index] += alpha * (
                    reward.total_hider_reward
                    + discount_factor * Q[next_state][next_action_index]
                    - Q[state][action_index]
                )

                if done:
                    break

                state = next_state
                action_index = next_action_index

        self.Q = Q
        self.stats = stats

    def get_action(self, state: GameState):
        action_index = epsilon_greedy(self.Q, state, train=False)
        action = {
            ag_id: PRODUCT_ACTIONS[action_index][ag_id - 1]
            for ag_id in SimpleAgent.ag_ids
        }

        return action


class DeterministicSeeker(SimpleAgent):
    def __init__(self, env_width, env_height):
        super(DeterministicSeeker, self).__init__()
        self._env_width = env_width
        self._env_height = env_height

        self._tactical_pos = GridPosition(self._env_width, self._env_height - 2)
        self._final_pos = GridPosition(self._env_width - 2, self._env_height - 4)

        self._tactical_pos_reached = False
        self._final_pos_reached = False

    def get_action(self, state: GameState):
        act1 = Action.NOP
        act2 = Action.NOP

        # get seeker positions
        seeker1_pos = state.seeker_positions[1]
        seeker2_pos = state.seeker_positions[2]

        if not self._tactical_pos_reached:
            # if seekers are not in the tactical position move them there

            if seeker1_pos != self._tactical_pos:
                # seeker 1 takes the north route
                if seeker1_pos.x == 1 and seeker1_pos.y != self._env_height:
                    # if seeker 1 is on the left shaft, move NORTH
                    act1 = Action.NORTH
                elif (
                    seeker1_pos.y == self._env_height
                    and seeker1_pos.x != self._env_width
                ):
                    # if seeker 1 is on the top shaft, EAST
                    act1 = Action.EAST
                elif (
                    seeker1_pos.x == self._env_width
                    and seeker1_pos != self._tactical_pos
                ):
                    # if seeker 1 is on the right shaft, SOUTH
                    act1 = Action.SOUTH

            if seeker2_pos != self._tactical_pos:
                # seeker 2 takes the south route
                if seeker2_pos.x == 1 and seeker2_pos.y != 1:
                    # if seeker 2 is on the left shaft, move NORTH
                    act2 = Action.SOUTH
                elif seeker2_pos.y == 1 and seeker2_pos.x != self._env_width:
                    # if seeker 2 is on the bottom shaft, move EAST
                    act2 = Action.EAST
                elif (
                    seeker2_pos.x == self._env_width
                    and seeker2_pos != self._tactical_pos
                ):
                    # if seeker 2 is on the right shaft, move NORTH
                    act2 = Action.NORTH

            if seeker1_pos == self._tactical_pos and seeker2_pos == self._tactical_pos:
                self._tactical_pos_reached = True

        else:
            if not self._final_pos_reached:
                # when seeker 1 and seeker 2 have reached the tactical position, move to the final one in tandem
                if seeker1_pos.y == self._tactical_pos.y and seeker1_pos.x != 3:
                    act1 = Action.WEST
                    act2 = Action.WEST
                elif seeker1_pos.x == 3 and seeker1_pos.y != self._final_pos.y:
                    act1 = Action.SOUTH
                    act2 = Action.SOUTH
                elif (
                    seeker1_pos.y == self._final_pos.y
                    and seeker1_pos.x != self._final_pos.x
                ):
                    act1 = Action.EAST
                    act2 = Action.EAST
                else:
                    self._final_pos_reached = True
            else:
                # if the agent made it thus far and the game has not ended, just make them move randomly
                return super(DeterministicSeeker, self).get_action(state)

        return {1: act1, 2: act2}
