import gym

from agents import PolicyIterationAgent, ValueIterationAgent, ValueIterationGSAgent, ValueIterationPSAgent

if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1")

    agent = ValueIterationPSAgent(env, gamma=0.9)

    observation = env.reset()
    for _ in range(1000):
        env.render()

        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()
