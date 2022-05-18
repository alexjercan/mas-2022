import numpy as np

EPSILON = 1e-5


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()


class PolicyIterationAgent:
    def __init__(self, env, gamma, max_iterations=10_000):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_prob = env.env.P

        self.values = np.zeros(self.num_states)
        self.policy = np.random.randint(
            0, self.num_actions, size=self.num_states
        ).astype(np.int64)

        self.updates = []

        self._policy_iteration()

    def _policy_iteration(self):
        for _ in range(self.max_iterations):
            prev_v = np.copy(self.values)

            for state in range(self.num_states):
                Q_value = 0
                action = self.policy[state]
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value += prob * (reward + self.gamma * prev_v[next_s])

                self.values[state] = Q_value

            for state in range(self.num_states):
                Q_value = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    for prob, next_s, reward, _ in self.state_prob[state][action]:
                        Q_value[action] += prob * (
                            reward + self.gamma * self.values[next_s]
                        )

                self.policy[state] = np.argmax(Q_value)

            self.updates.append(
                np.sqrt(np.sum((self.values - prev_v) * (self.values - prev_v)))
            )
            if np.max(np.abs(self.values - prev_v)) < EPSILON:
                break

    def choose_action(self, observation):
        return self.policy[observation]

    def __str__(self):
        return "PolicyIteration"


class ValueIterationAgent:
    def __init__(self, env, gamma, max_iterations=10_000):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_prob = env.env.P

        self.values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states).astype(np.int64)

        self.updates = []

        self._value_iteration()
        self._extract_policy()

    def _value_iteration(self):
        for _ in range(self.max_iterations):
            prev_v = np.copy(self.values)

            for state in range(self.num_states):
                Q_value = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    for prob, next_s, reward, _ in self.state_prob[state][action]:
                        Q_value[action] += prob * (reward + self.gamma * prev_v[next_s])

                self.values[state] = np.max(Q_value)

            self.updates.append(
                np.sqrt(np.sum((self.values - prev_v) * (self.values - prev_v)))
            )
            if np.max(np.abs(self.values - prev_v)) < EPSILON:
                break

    def _extract_policy(self):
        for state in range(self.num_states):
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * self.values[next_s]
                    )

            self.policy[state] = np.argmax(Q_value)

    def choose_action(self, observation):
        return self.policy[observation]

    def __str__(self):
        return "ValueIteration"


class ValueIterationGSAgent:
    def __init__(self, env, gamma, max_iterations=10_000):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_prob = env.env.P

        self.values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states).astype(np.int64)

        self.updates = []

        self._value_iteration()
        self._extract_policy()

    def _value_iteration(self):
        for _ in range(self.max_iterations):
            prev_v = np.copy(self.values)

            for state in range(self.num_states):
                Q_value = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    for prob, next_s, reward, _ in self.state_prob[state][action]:
                        Q_value[action] += prob * (
                            reward + self.gamma * self.values[next_s]
                        )

                self.values[state] = np.max(Q_value)

            self.updates.append(
                np.sqrt(np.sum((self.values - prev_v) * (self.values - prev_v)))
            )
            if np.max(np.abs(self.values - prev_v)) < EPSILON:
                break

    def _extract_policy(self):
        for state in range(self.num_states):
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * self.values[next_s]
                    )

            self.policy[state] = np.argmax(Q_value)

    def choose_action(self, observation):
        return self.policy[observation]

    def __str__(self):
        return "ValueIterationGS"


class ValueIterationPSAgent:
    def __init__(self, env, gamma, max_iterations=10_000):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_prob = env.env.P

        self.values = np.zeros(self.num_states)
        self.priority = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states).astype(np.int64)

        self.updates = []


        # update the priority
        #  1. Compute the future values
        future_v = np.copy(self.values)

        for state in range(self.num_states):
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * future_v[next_s]
                    )

            future_v[state] = np.max(Q_value)

        # 2. Compute the error w.r.t the future value
        self.priority = np.abs(future_v - self.values)


        self._value_iteration()
        self._extract_policy()

    def _value_iteration(self):
        for i in range(self.max_iterations):
            prev_v = np.copy(self.values)

            # get the state with the maximum priority
            state = np.argmax(self.priority)

            # compute the value of the selected state
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * self.values[next_s]
                    )

            self.values[state] = np.max(Q_value)

            # update the priority
            #  1. Compute the future values
            future_v = np.copy(self.values)

            for state in range(self.num_states):
                Q_value = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    for prob, next_s, reward, _ in self.state_prob[state][action]:
                        Q_value[action] += prob * (
                            reward + self.gamma * future_v[next_s]
                        )

                future_v[state] = np.max(Q_value)

            # 2. Compute the error w.r.t the future value
            self.priority = np.abs(future_v - self.values)

            # Check if converged
            self.updates.append(
                np.sqrt(np.sum((self.values - prev_v) * (self.values - prev_v)))
            )
            if np.max(np.abs(self.values - prev_v)) < EPSILON:
                break


    def _extract_policy(self):
        for state in range(self.num_states):
            Q_value = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for prob, next_s, reward, _ in self.state_prob[state][action]:
                    Q_value[action] += prob * (
                        reward + self.gamma * self.values[next_s]
                    )

            self.policy[state] = np.argmax(Q_value)

    def choose_action(self, observation):
        return self.policy[observation]

    def __str__(self):
        return "ValueIterationPS"
