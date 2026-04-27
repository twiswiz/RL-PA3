import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class SensorTransmissionEnv(gym.Env):
    """
    State:
        state = [theta_idx, battery, estimate_idx, max_idx]

        theta_idx    : index i such that theta_t = 0.02 * i
        battery      : battery level b_t
        estimate_idx : index of central station estimate before current slot update
        max_idx      : index of max pollution since last successful transmission,
                       including current theta_t

    Actions:
        0 : do not transmit
        1 : transmit true pollution theta_t
        2 : transmit max pollution since last successful transmission
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # System parameters from assignment
        self.lam = 0.1
        self.B = 10
        self.eta = 2
        self.Delta = 3

        # Pollution space: {0, 0.02, ..., 0.98, 1}
        self.num_pollution_states = 51
        self.pollution_values = np.linspace(0.0, 1.0, self.num_pollution_states)

        # Load transition matrix and solar probability vector
        base_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

        self.P = np.load(base_path / "air.npy").astype(np.float64)
        self.alpha = np.load(base_path / "solar.npy").astype(np.float64)

        # Basic checks
        if self.P.shape != (self.num_pollution_states, self.num_pollution_states):
            raise ValueError("air.npy must have shape (51, 51).")

        if self.alpha.shape[0] != self.Delta + 1:
            raise ValueError("solar.npy must have length 4 because Delta = 3.")

        # Normalize safely in case of tiny floating point errors
        self.P = self.P / self.P.sum(axis=1, keepdims=True)
        self.alpha = self.alpha / self.alpha.sum()

        # Observation space:
        # theta_idx in {0,...,50}
        # battery in {0,...,10}
        # estimate_idx in {0,...,50}
        # max_idx in {0,...,50}
        self.observation_space = spaces.MultiDiscrete(
            np.array([
                self.num_pollution_states,
                self.B + 1,
                self.num_pollution_states,
                self.num_pollution_states
            ], dtype=np.int64)
        )

        # Gymnasium action space cannot directly be state-dependent.
        # So we define all 3 actions globally.
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.time_slot = 0
        self.max_time_slots = 288

    def _loss(self, theta_idx, estimate_idx):
        """
        Loss is computed after the transmission attempt in the current time slot.

        If theta <= estimate:
            loss = |theta - estimate|
        If theta > estimate:
            loss = 1.5 * |theta - estimate|
        """
        theta = self.pollution_values[theta_idx]
        estimate = self.pollution_values[estimate_idx]

        diff = abs(theta - estimate)

        if theta <= estimate:
            return diff
        else:
            return 1.5 * diff

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time_slot = 0

        # Three states uniformly at random:
        theta_idx = self.np_random.integers(0, self.num_pollution_states)
        battery = self.np_random.integers(0, self.B + 1)
        estimate_idx = self.np_random.integers(0, self.num_pollution_states)

        # Fourth state:
        # At the beginning of an episode, no previous unreported pollution history exists.
        # So max since last successful transmission is initialized to current theta.
        max_idx = theta_idx

        self.state = np.array(
            [theta_idx, battery, estimate_idx, max_idx],
            dtype=np.int64
        )

        return self.state.copy(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before calling step().")

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("Invalid action. Valid actions are 0, 1, and 2.")

        theta_idx, battery, estimate_idx, max_idx = self.state

        # If battery is insufficient, transmission actions are not actually possible.
        # We treat such an attempted transmission as no transmission.
        if action in [1, 2] and battery < self.eta:
            action = 0

        success = False
        new_estimate_idx = estimate_idx

        # Battery after possible transmission cost
        battery_after_action = battery

        if action == 0:
            # No transmission
            new_estimate_idx = estimate_idx
            success = False

        elif action == 1:
            # Transmit true pollution theta_t
            battery_after_action = battery - self.eta

            success = self.np_random.random() < self.lam

            if success:
                new_estimate_idx = theta_idx
            else:
                new_estimate_idx = estimate_idx

        elif action == 2:
            # Transmit maximum pollution since last successful transmission
            battery_after_action = battery - self.eta

            success = self.np_random.random() < self.lam

            if success:
                new_estimate_idx = max_idx
            else:
                new_estimate_idx = estimate_idx

        # Reward is negative loss because the objective is to minimize discounted loss
        current_loss = self._loss(theta_idx, new_estimate_idx)
        reward = -current_loss

        # Sample solar energy delta_t
        solar_energy = self.np_random.choice(
            np.arange(self.Delta + 1),
            p=self.alpha
        )

        # Update battery
        next_battery = min(self.B, battery_after_action + solar_energy)

        # Sample next pollution theta_{t+1}
        next_theta_idx = self.np_random.choice(
            np.arange(self.num_pollution_states),
            p=self.P[theta_idx]
        )

        # Update max pollution memory
        if success:
            # Successful transmission resets the accumulated max history.
            next_max_idx = next_theta_idx
        else:
            # No successful update, so keep accumulating the maximum.
            next_max_idx = max(max_idx, next_theta_idx)

        # Next state:
        # estimate_idx for next slot is the estimate after current slot transmission.
        self.state = np.array(
            [next_theta_idx, next_battery, new_estimate_idx, next_max_idx],
            dtype=np.int64
        )

        self.time_slot += 1

        terminated = False
        truncated = self.time_slot >= self.max_time_slots
        info = {}

        return self.state.copy(), reward, terminated, truncated, info