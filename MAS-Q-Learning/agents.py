import numpy as np

from tqdm import tqdm

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, epochs, eval_episodes=50, verbose=True):
        # initialize the agent
        self.env = env
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.epochs = epochs # number of epochs
        self.eval_episodes = eval_episodes # number of episodes to evaluate the agent
        self.verbose = verbose # verbosity

        # initialize the Q-table
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def update(self, state, action, reward, next_state):
        # update the Q-table
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])

    def act(self, state):
        # choose an action based on the current state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def train(self):
        rewards, policies = [], []
        pbar = tqdm(range(self.epochs)) if self.verbose else range(self.epochs)

        # train the agent
        for epoch in pbar:
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

            # evaluate the agent if eval_episodes is reached
            if epoch % self.eval_episodes == 0:
                policy = np.argmax(self.Q, axis=1)
                rewards.append(self.test(policy))
                policies.append(policy)

        # return the rewards and the best policy
        return rewards, policies[np.argmax(rewards)]

    def test(self, policy):
        # test the agent
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy[state]
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

        return total_reward


class SarsaAgent:
    def __init__(self, env, alpha, gamma, epsilon, epochs, eval_episodes=50, verbose=True):
        # initialize the agent
        self.env = env
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.epochs = epochs # number of epochs
        self.eval_episodes = eval_episodes # number of episodes to evaluate the agent
        self.verbose = verbose # verbosity

        # initialize the Q-table
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def update(self, state, action, reward, next_state, next_action):
        # update the Q-table
        self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

    def act(self, state):
        # choose an action based on the current state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def train(self):
        rewards, policies = [], []
        pbar = tqdm(range(self.epochs)) if self.verbose else range(self.epochs)

        # train the agent
        for epoch in pbar:
            state = self.env.reset()
            action = self.act(state)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.act(next_state)
                self.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action

            # evaluate the agent if eval_episodes is reached
            if epoch % self.eval_episodes == 0:
                policy = np.argmax(self.Q, axis=1)
                rewards.append(self.test(policy))
                policies.append(policy)

        # return the rewards and the best policy
        return rewards, policies[np.argmax(rewards)]

    def test(self, policy):
        # test the agent
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy[state]
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

        return total_reward
