import torch

import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def featurize(state: np.ndarray):
    return torch.from_numpy(state).float().unsqueeze(0)


class QLearningAgent:
    def __init__(self, env, model, learning_rate, gamma, epsilon, epsilon_decay, episodes, max_steps):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.max_steps = max_steps

        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def act(self, state, train=True):
        if np.random.rand() < self.epsilon and train:
            return self.env.action_space.sample()

        x_state = featurize(state)
        q_values = self.model(x_state)[0]
        return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        x_state = featurize(state)
        x_next_state = featurize(next_state)

        if done:
            target_q_value = torch.tensor(reward)
        else:
            target_q_value = reward + self.gamma * self.model(x_next_state)[0].max().detach()

        current_q_value = self.model(x_state)[0][action]

        loss = F.mse_loss(current_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        rewards = []

        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if done:
                    break

            self.epsilon *= self.epsilon_decay
            rewards.append(episode_reward)

        return rewards
