import random

import numpy as np


class QLearningEtaOptimizer:
    def __init__(self, initial_eta=0.01):
        self.initial_eta = initial_eta
        self.current_eta = initial_eta

        self.alpha = 0.5
        self.gamma = 0.7
        self.epsilon = 0.7
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.1

        self.time_bins = np.array([
            0,
            1 * 24 * 60,
            2 * 24 * 60,
            3 * 24 * 60,
            10 * 24 * 60,
        ])
        self.time = None

        # Actions: 0=maintain eta, 1=decrease eta 0.9, 2=decrease eta 0.75
        # 3=decrease eta 0.5, 4=decrease eta 0.25, 5=decrease eta 0.1
        self.actions = [0, 1, 2, 3, 4, 5]

        self.state_action = [False] * (self.time_bins.shape[0] - 1)
        self.first_run = True
        self.states_actions = []
        self.etas = []

        self.q_table = np.zeros((21, 6))

    def update_time(self, time: float) -> None:
        self.time = time

    def get_state(self) -> int:
        """Convert current eta to a state index for the Q-table"""
        if self.first_run:
            self.first_run = False
            return 0

        state_day = np.digitize(self.time, self.time_bins)
        state_day = max(min(state_day, self.time_bins.shape[0] - 1), 1) - 1

        if self.current_eta >= 0.005 and self.current_eta <= 0.01:
            state_eta = 1
        elif self.current_eta >= 0.001 and self.current_eta < 0.005:
            state_eta = 2
        elif self.current_eta >= 0.0005 and self.current_eta < 0.001:
            state_eta = 3
        elif self.current_eta >= 0.0001 and self.current_eta < 0.0005:
            state_eta = 4
        elif self.current_eta >= 0 and self.current_eta < 0.0001:
            state_eta = 5

        state = (state_day - 1) * 5 + state_eta
        return state

    def choose_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def update_eta(self, action: int) -> float:
        """Apply the selected action to update eta"""
        if action == 0:  # Maintain 1.0
            self.current_eta *= 1.0
        elif action == 1:  # Decrease 0.9
            self.current_eta *= 0.9
        elif action == 2:  # Decrease 0.75
            self.current_eta *= 0.75
        elif action == 3:  # Decrease 0.5
            self.current_eta *= 0.5
        elif action == 4:  # Decrease 0.25
            self.current_eta *= 0.25
        elif action == 5:  # Decrease 0.1
            self.current_eta *= 0.1

        return self.current_eta

    def update_q_table(
        self, state: int, action: int, reward: float, next_state: int
    ) -> None:
        if next_state is None:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = (
                reward
                + self.gamma * self.q_table[next_state][best_next_action]
            )
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def should_run(self) -> bool:
        if self.time >= self.time_bins[0] and self.time < self.time_bins[1]:
            if self.state_action[0] is False:
                self.state_action[0] = True
                return True
        elif self.time >= self.time_bins[1] and self.time < self.time_bins[2]:
            if self.state_action[1] is False:
                self.state_action[1] = True
                return True
        elif self.time >= self.time_bins[2] and self.time < self.time_bins[3]:
            if self.state_action[2] is False:
                self.state_action[2] = True
                return True
        elif self.time >= self.time_bins[3]:
            if self.state_action[3] is False:
                self.state_action[3] = True
                return True

        return False

    def run(self, time: float) -> float:
        self.update_time(time)
        if not self.should_run():
            return self.current_eta

        state = self.get_state()
        action = self.choose_action(int(state))
        new_eta = self.update_eta(action)
        self.current_eta = new_eta
        self.etas.append(new_eta)

        if len(self.states_actions) > 0:
            self.states_actions[-1][-1] = int(state)
        self.states_actions.append([int(state), int(action), None])

        self.decay_exploration()

        return new_eta

    def reset_state_actions(self) -> None:
        self.state_action = [False] * (self.time_bins.shape[0] - 1)
        self.states_actions = []
        self.first_run = True
        self.etas = []

    def reset_eta(self, initial_eta: float) -> None:
        self.current_eta = initial_eta

    def update_q_table_episode(self, metrics: dict) -> None:
        if metrics["tir"] < 0.70:
            reward = -1
        elif metrics["tir"] < 0.80:
            reward = +1
        elif metrics["tir"] < 0.90:
            reward = +2
        else:
            reward = +3

        if metrics["tir"] > 0.70:
            if metrics["bgc_min"] < 70:
                if metrics["bgc_min"] < 54:
                    reward += -1
                else:
                    reward += -0.5
            else:
                reward += +2

        self.current_reward = reward
        for state, action, next_state in self.states_actions:
            self.update_q_table(state, action, reward, next_state)

    def get_best_actions_etas(self) -> dict:
        actions = []
        states = []
        etas = []

        eta = self.initial_eta
        state = 0
        for day in [1, 2, 3, 4]:
            action = np.argmax(self.q_table[state])
            if action == 0:
                eta *= 1.0
            elif action == 1:
                eta *= 0.9
            elif action == 2:
                eta *= 0.75
            elif action == 3:
                eta *= 0.5
            elif action == 4:
                eta *= 0.25
            elif action == 5:
                eta *= 0.1

            actions.append(int(action))
            etas.append(eta)
            states.append(state)

            state_day = day
            if eta >= 0.005 and eta <= 0.01:
                state_eta = 1
            elif eta >= 0.001 and eta < 0.005:
                state_eta = 2
            elif eta >= 0.0005 and eta < 0.001:
                state_eta = 3
            elif eta >= 0.0001 and eta < 0.0005:
                state_eta = 4
            elif eta >= 0 and eta < 0.0001:
                state_eta = 5
            state = (state_day - 1) * 5 + state_eta

        states.append(state)

        return {"actions": actions, "states": states, "etas": etas}
