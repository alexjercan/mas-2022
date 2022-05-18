import itertools
import concurrent.futures
import numpy as np
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm
# from gym.envs.toy_text import discrete
from io import StringIO
from collections import defaultdict


EPSILON = 1e-8


class CyclicBuffer:
    def __init__(self, n, dtype=np.float32):
        self.n = n
        self.buffer = np.zeros(n, dtype=dtype)
        self.index = 0

    def add(self, x):
        self.buffer[self.index] = x
        self.index = (self.index + 1) % self.n

    def get(self):
        return self.buffer[self.index]

    def __getitem__(self, key):
        key = (self.index + key) % self.n
        return self.buffer[key]

    def __setitem__(self, key, value):
        key = (self.index + key) % self.n
        self.buffer[key] = value

    def __len__(self):
        return self.n

    def __str__(self):
        return str(self.buffer)


class NStepSARSAAgent:
    def __init__(self, env, n, gamma, alpha, epsilon=0.1, episodes=200):
        self.env = env  # The environment
        self.n = n  # The number of steps to take
        self.gamma = gamma  # The discount factor
        self.alpha = alpha  # The learning rate
        self.epsilon = epsilon  # The probability of a random action
        self.episodes = episodes  # The number of episodes to run

        self.Q = np.random.rand(env.observation_space.n, env.action_space.n)

        self.actions = CyclicBuffer(n, dtype=np.int32)
        self.states = CyclicBuffer(n, dtype=np.int32)
        self.rewards = CyclicBuffer(n, dtype=np.float32)

    def act(self, state, train=True):
        if np.random.random() < self.epsilon and train:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def train(self):
        rewards = []

        for _ in range(self.episodes):
            episode_reward = 0

            self.actions = CyclicBuffer(self.n, dtype=np.int32)
            self.states = CyclicBuffer(self.n, dtype=np.int32)
            self.rewards = CyclicBuffer(self.n, dtype=np.float32)

            state = self.env.reset()
            self.states[0] = state

            action = self.act(state)
            self.actions[0] = action

            T = float("inf")

            t = 0
            while True:
                if t < T:
                    next_state, reward, done, _ = self.env.step(self.actions[t])
                    episode_reward += reward

                    self.states[t + 1] = next_state
                    self.rewards[t + 1] = reward

                    if done:
                        T = t + 1
                    else:
                        action = self.act(next_state)
                        self.actions[t + 1] = action

                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += self.gamma ** (i - tau - 1) * self.rewards[i]
                    self.Q[self.states[tau], self.actions[tau]] += self.alpha * (
                        G - self.Q[self.states[tau], self.actions[tau]]
                    )

                if tau == T - 1:
                    break

                t += 1

            rewards.append(episode_reward)

        return rewards


class ValueIterationAgent:
    def __init__(self, env, gamma, max_iterations=10_000):
        self.max_iterations = max_iterations  # The maximum number of iterations to run
        self.gamma = gamma  # The discount factor
        self.num_states = env.observation_space.n  # The number of states
        self.num_actions = env.action_space.n  # The number of actions
        self.state_prob = env.env.P  # The transition probabilities

        self.values = np.zeros(self.num_states)  # The values of each state
        self.policy = np.zeros(self.num_states).astype(
            np.int64
        )  # The policy of each state

    def train(self):
        for _ in range(self.max_iterations):
            prev_v = np.copy(self.values)

            for state in range(self.num_states):
                Q_value = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    for prob, next_s, reward, _ in self.state_prob[state][action]:
                        Q_value[action] += prob * (reward + self.gamma * prev_v[next_s])

                self.values[state] = np.max(Q_value)

            if np.max(np.abs(self.values - prev_v)) < EPSILON:
                break

        for state in range(self.num_states):
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * self.values[next_s]
                    )

            self.policy[state] = np.argmax(Q_value)

    def act(self, observation):
        return self.policy[observation]


def rmse_fn(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv():
    metadata = {"render.modes": ["human", "ansi"]}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = (
            np.array(current)
            + np.array(delta)
            + np.array([-1, 0]) * winds[tuple(current)]
        )
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:, [3, 4, 5, 8]] = 1
        winds[:, [6, 7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode="human", close=False):
        self._render(mode, close)

    def _render(self, mode="human", close=False):
        if close:
            return

        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3, 7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")


def create_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA) * (epsilon / nA)
        best_action = np.argmax(Q[observation])
        A[best_action] += 1.0 - epsilon
        return A

    return policy_fn


def n_step_sarsa(env, num_episodes, n=5, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    NOTE: some parts taken from https://github.com/Breakend/MultiStepBootstrappingInRL/blob/master/n_step_sarsa.py
    """

    # The final action-value function.
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The policy we're following
    policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    max_reward = 0
    total_reward = 0
    rewards_per_episode = []

    for i_episode in range(num_episodes):
        # initializations
        T = sys.maxsize
        tau = 0
        t = -1
        stored_actions = {}
        stored_rewards = {}
        stored_states = {}

        # initialize first state
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(env.action_space.n, p=action_probs)

        stored_actions[0] = action
        stored_states[0] = state
        reward_for_episode = 0

        while tau < (T - 1):
            t += 1
            if t < T:
                state, reward, done, _ = env.step(action)

                stored_rewards[(t + 1) % n] = reward
                stored_states[(t + 1) % n] = state

                total_reward += reward
                reward_for_episode += reward

                if done:
                    T = t + 1
                else:
                    next_action_probs = policy(state)
                    action = np.random.choice(env.action_space.n, p=next_action_probs)
                    stored_actions[(t + 1) % n] = action
            tau = t - n + 1

            if tau >= 0:
                # calculate G(tau:tau+n)
                G = np.sum(
                    [
                        discount_factor ** (i - tau - 1) * stored_rewards[i % n]
                        for i in range(tau + 1, min(tau + n, T) + 1)
                    ]
                )

                if tau + n < T:
                    G += (
                        discount_factor ** n
                        * Q[stored_states[(tau + n) % n]][stored_actions[(tau + n) % n]]
                    )

                tau_s, tau_a = stored_states[tau % n], stored_actions[tau % n]

                # update Q value with n step return
                Q[tau_s][tau_a] += alpha * (G - Q[tau_s][tau_a])

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)

    return Q, max_reward


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(s, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    it = 1
    while True:
        # stopping condition
        delta = 0
        # sweeping thru all the states
        for s in range(env.nS):
            v = V[s]
            # get the action values of current state
            action_values = one_step_lookahead(s, V)

            # get the best action value
            best_action_value = np.max(action_values)

            # update value function
            V[s] = best_action_value

            delta = np.maximum(delta, np.abs(v - V[s]))

        if delta < theta:
            break
        it += 1
    # create a deterministic policy
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # one step lookahead
        action_values = one_step_lookahead(s, V)
        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0

    return policy, V


if __name__ == "__main__":
    # Value Iteration
    env = WindyGridworldEnv()

    policy, V = value_iteration(env, discount_factor=0.99)

    gt_values = V

    # N Step Sarsa
    LEARNING_RATES = np.arange(0.1, 1.1, 0.1)
    N_VALUES = [2, 4, 8, 16]
    DISCOUNT_FACTOR = 0.99

    combs = list(itertools.product(LEARNING_RATES, N_VALUES))
    results = defaultdict(list)
    with tqdm(total=len(combs)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            future_to_n_value = {
                executor.submit(
                    n_step_sarsa, WindyGridworldEnv(), 200, n, DISCOUNT_FACTOR, lr
                ): (n, lr)
                for (lr, n) in combs
            }

            for future in concurrent.futures.as_completed(future_to_n_value):
                n, lr = future_to_n_value[future]

                try:
                    Q, max_reward = future.result()
                    values = np.max(Q, axis=1)
                    rmse = rmse_fn(values, gt_values)
                    results[n].append((lr, rmse))
                except Exception as exc:
                    print(exc)
                else:
                    pbar.update(1)

    for n in N_VALUES:
        plt.plot(*zip(*results[n]), label=f"{n}")

    plt.xlabel(f"Learning rate")
    plt.ylabel("RMSE")
    plt.title("RMSE Between VI and N-SARSA")

    plt.legend()
    plt.show()
