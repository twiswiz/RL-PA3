import numpy as np
import matplotlib.pyplot as plt
from GymAirQuality import SensorTransmissionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_valid_actions(battery, eta):
    if battery < eta:
        return [0]
    return [0, 1, 2]


# ---------------------------------------------------------------------------
# Testing(Q) — follows pseudocode exactly
#   - Creates a NEW environment
#   - Nep episodes with epsilon = 0 (pure greedy)
#   - Returns avg_reward / Nep
# ---------------------------------------------------------------------------

def Testing(Q, Nep=50):
    env = SensorTransmissionEnv()          # create new env as per pseudocode
    avg_reward = 0.0

    for ep in range(Nep):
        state, _ = env.reset()
        ep_reward = 0.0

        while True:
            theta_idx, battery, estimate_idx, max_idx = map(int, state)
            valid = get_valid_actions(battery, env.eta)

            # choose_action(Q, x, epsilon=0) — pure greedy, no exploration
            q_vals = Q[theta_idx, battery, estimate_idx, max_idx, valid]
            action = valid[int(np.argmax(q_vals))]

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)

            if terminated or truncated:
                break

        avg_reward += ep_reward

    env.close()
    return avg_reward / Nep


# ---------------------------------------------------------------------------
# Load policies and reconstruct Q-table shape for policy evaluation
# We use the policy directly (greedy by definition) instead of Q-table
# ---------------------------------------------------------------------------

def Testing_from_policy(policy, Nep=50):
    """
    Same as Testing(Q) but takes the saved policy array directly,
    since we saved policy (not Q) to .npy files.
    """
    env = SensorTransmissionEnv()
    avg_reward = 0.0

    for ep in range(Nep):
        state, _ = env.reset()
        ep_reward = 0.0

        while True:
            theta_idx, battery, estimate_idx, max_idx = map(int, state)

            # Policy already encodes the greedy action; respect valid actions
            action = int(policy[theta_idx, battery, estimate_idx, max_idx])

            # Safety: if policy gives invalid action due to battery, fall back
            if action in [1, 2] and battery < env.eta:
                action = 0

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)

            if terminated or truncated:
                break

        avg_reward += ep_reward

    env.close()
    return avg_reward / Nep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    Nep = 50   # number of test episodes as specified

    # ---- Load saved policies ----------------------------------------------
    policy1 = np.load("policy1.npy")   # standard Q-Learning
    policy2 = np.load("policy2.npy")   # structural knowledge Q-Learning

    print(f"Testing both policies over {Nep} episodes (epsilon = 0)...\n")

    avg1 = Testing_from_policy(policy1, Nep=Nep)
    avg2 = Testing_from_policy(policy2, Nep=Nep)

    print(f"  Q-Learning policy       avg reward / Nep = {avg1:.4f}")
    print(f"  Structural Knowledge    avg reward / Nep = {avg2:.4f}")
    print(f"  Improvement (SK - QL)                   = {avg2 - avg1:.4f}")

    # ---- Plot training curves from saved test rewards ---------------------
    tr_ql = np.load("test_rewards_ql.npy")   # shape (N, 2): [episode, avg_reward]
    tr_sk = np.load("test_rewards_sk.npy")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tr_ql[:, 0], tr_ql[:, 1], label="Q-Learning",
            color="steelblue", linewidth=1.5)
    ax.plot(tr_sk[:, 0], tr_sk[:, 1], label="Structural Knowledge Q-Learning",
            color="darkorange", linewidth=1.5)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Greedy Return", fontsize=12)
    ax.set_title("Average Greedy Return vs Episode", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("\nPlot saved to training_curves.png")