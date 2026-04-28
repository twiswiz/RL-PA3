import numpy as np
import gymnasium as gym
from GymAirQuality import SensorTransmissionEnv

def get_valid_actions(battery, eta):
    if battery < eta:
        return [0]
    return [0, 1, 2]


def greedy_episode(env, Q):
    state, _ = env.reset()
    total_reward = 0.0

    while True:
        theta_idx, battery, estimate_idx, max_idx = map(int, state)
        valid = get_valid_actions(battery, env.eta)

        q_vals = Q[theta_idx, battery, estimate_idx, max_idx, valid]
        action = valid[int(np.argmax(q_vals))]

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    return total_reward


def evaluate_greedy_policy(env, Q, n_test_episodes=20):
    rewards = [greedy_episode(env, Q) for _ in range(n_test_episodes)]
    return float(np.mean(rewards))


def q_learning_update(Q, state, action, reward, next_state, beta, alpha, eta):
    theta_idx, battery, estimate_idx, max_idx = map(int, state)
    nt, nb, ne, nm = map(int, next_state)

    next_valid = get_valid_actions(nb, eta)
    max_next_q = float(np.max(Q[nt, nb, ne, nm, next_valid]))

    td_target = reward + beta * max_next_q
    td_error = td_target - Q[theta_idx, battery, estimate_idx, max_idx, action]
    Q[theta_idx, battery, estimate_idx, max_idx, action] += alpha * td_error


def extract_policy(Q, env):
    n_theta = env.num_pollution_states
    n_battery = env.B + 1

    policy = np.zeros((n_theta, n_battery, n_theta, n_theta), dtype=np.int64)

    for ti in range(n_theta):
        for bi in range(n_battery):
            valid = get_valid_actions(bi, env.eta)
            for ei in range(n_theta):
                for mi in range(n_theta):
                    q_vals = Q[ti, bi, ei, mi, valid]
                    policy[ti, bi, ei, mi] = valid[int(np.argmax(q_vals))]

    return policy

def QLearning(env, beta, Nepisodes, alpha):
    n_theta = env.num_pollution_states
    n_battery = env.B + 1
    n_actions = env.action_space.n

    Q = np.zeros((n_theta, n_battery, n_theta, n_theta, n_actions), dtype=np.float64)

    epsilon_start = 1.0
    epsilon_end = 0.05
    decay_until = max(1, int(0.8 * Nepisodes))

    def epsilon(ep):
        if ep >= decay_until:
            return epsilon_end
        frac = ep / decay_until
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    TEST_FREQ = 50
    test_rewards = []

    for ep in range(Nepisodes):
        state, _ = env.reset()
        eps = epsilon(ep)

        while True:
            theta_idx, battery, estimate_idx, max_idx = map(int, state)
            valid = get_valid_actions(battery, env.eta)

            if np.random.random() < eps:
                action = int(np.random.choice(valid))
            else:
                q_vals = Q[theta_idx, battery, estimate_idx, max_idx, valid]
                action = valid[int(np.argmax(q_vals))]

            next_state, reward, terminated, truncated, _ = env.step(action)
            q_learning_update(Q, state, action, reward, next_state, beta, alpha, env.eta)

            state = next_state
            if terminated or truncated:
                break

        if ep % TEST_FREQ == 0:
            test_reward = evaluate_greedy_policy(env, Q)
            test_rewards.append((ep, test_reward))
            if ep % 500 == 0:
                print(f"  QL episode {ep:5d}/{Nepisodes}  eps={eps:.3f}  "
                      f"avg greedy return = {test_reward:.4f}")

    np.save("test_rewards_ql.npy", np.array(test_rewards, dtype=np.float64))
    print("Q-Learning training complete. Test rewards saved to test_rewards_ql.npy")

    return extract_policy(Q, env)

def _action_cost(action, eta):
    return 0 if int(action) == 0 else int(eta)


def _infer_unique_solar_delta(env, state, action, next_state):
    _, battery, _, _ = map(int, state)
    _, next_battery, _, _ = map(int, next_state)
    cost = _action_cost(action, env.eta)

    candidates = []
    for delta in range(env.Delta + 1):
        predicted = min(env.B, battery - cost + delta)
        if predicted == next_battery:
            candidates.append(delta)

    if len(candidates) != 1:
        return None
    return candidates[0]


def _infer_transmission_outcome(env, state, action, next_state):
    if action == 0:
        return "no_transmit"

    theta, battery, estimate, max_seen = map(int, state)
    next_theta, next_battery, next_estimate, next_max_seen = map(int, next_state)

    transmitted_value = theta if action == 1 else max_seen

    failure_match = (
        next_estimate == estimate and
        next_max_seen == max(max_seen, next_theta)
    )

    success_match = (
        next_estimate == transmitted_value and
        next_max_seen == next_theta
    )

    if success_match and not failure_match:
        return "success"
    if failure_match and not success_match:
        return "failure"

    # Ambiguous case, e.g. transmitted value already equals old estimate.
    return None


def _pseudo_transition(env, real_state, real_action, real_next_state,
                       target_state, target_action, solar_delta, outcome):
    theta1, battery1, estimate1, max1 = map(int, real_state)
    next_theta1, _, _, _ = map(int, real_next_state)

    theta2, battery2, estimate2, max2 = map(int, target_state)
    target_action = int(target_action)

    if theta2 != theta1:
        return None

    if target_action not in get_valid_actions(battery2, env.eta):
        return None

    if max2 < theta2:
        return None

    if real_action == 0 and target_action != 0:
        return None

    if target_action == 0:
        next_estimate2 = estimate2
        next_max2 = max(max2, next_theta1)
        next_battery2 = min(env.B, battery2 + solar_delta)
        pseudo_reward = -env._loss(theta1, next_estimate2)
        pseudo_next_state = np.array(
            [next_theta1, next_battery2, next_estimate2, next_max2],
            dtype=np.int64,
        )
        return pseudo_reward, pseudo_next_state

    if outcome not in ("success", "failure"):
        return None

    next_battery2 = min(env.B, battery2 - env.eta + solar_delta)

    if outcome == "failure":
        next_estimate2 = estimate2
        next_max2 = max(max2, next_theta1)
        pseudo_reward = -env._loss(theta1, next_estimate2)
    else:
        if target_action == 1:
            transmitted_value2 = theta1       # transmit true pollution
        else:
            transmitted_value2 = max2         # transmit max since last success

        next_estimate2 = transmitted_value2
        next_max2 = next_theta1               # reset history after success
        pseudo_reward = -env._loss(theta1, next_estimate2)

    pseudo_next_state = np.array(
        [next_theta1, next_battery2, next_estimate2, next_max2],
        dtype=np.int64,
    )
    return pseudo_reward, pseudo_next_state


def _sample_structural_target_states(env, theta_idx, rng, n_targets):
    targets = []
    for _ in range(n_targets):
        battery2 = int(rng.integers(0, env.B + 1))
        estimate2 = int(rng.integers(0, env.num_pollution_states))
        # Reachable states satisfy max_idx >= theta_idx.
        max2 = int(rng.integers(theta_idx, env.num_pollution_states))
        targets.append(np.array([theta_idx, battery2, estimate2, max2], dtype=np.int64))
    return targets


def QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha):

    n_theta = env.num_pollution_states
    n_battery = env.B + 1
    n_actions = env.action_space.n

    Q = np.zeros((n_theta, n_battery, n_theta, n_theta, n_actions), dtype=np.float64)

    epsilon_start = 1.0
    epsilon_end = 0.05
    decay_until = max(1, int(0.8 * Nepisodes))

    def epsilon(ep):
        if ep >= decay_until:
            return epsilon_end
        frac = ep / decay_until
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    STRUCTURAL_TARGETS_PER_STEP = 4

    rng = np.random.default_rng()

    TEST_FREQ = 50
    test_rewards = []

    for ep in range(Nepisodes):
        state, _ = env.reset()
        eps = epsilon(ep)

        while True:
            theta_idx, battery, estimate_idx, max_idx = map(int, state)
            valid = get_valid_actions(battery, env.eta)

            if np.random.random() < eps:
                action = int(np.random.choice(valid))
            else:
                q_vals = Q[theta_idx, battery, estimate_idx, max_idx, valid]
                action = valid[int(np.argmax(q_vals))]

            next_state, reward, terminated, truncated, _ = env.step(action)

            # 1. Normal Q-learning update on the actual sample.
            q_learning_update(Q, state, action, reward, next_state, beta, alpha, env.eta)

            # 2. Structural-knowledge pseudo-updates.
            solar_delta = _infer_unique_solar_delta(env, state, action, next_state)

            if solar_delta is not None:
                outcome = _infer_transmission_outcome(env, state, action, next_state)

                # Include the current state as a target too, but actual pair is
                # skipped below to avoid doing the exact same update twice.
                target_states = [np.array(state, dtype=np.int64)]
                target_states.extend(
                    _sample_structural_target_states(
                        env, theta_idx, rng, STRUCTURAL_TARGETS_PER_STEP
                    )
                )

                for target_state in target_states:
                    _, target_battery, _, _ = map(int, target_state)
                    target_valid_actions = get_valid_actions(target_battery, env.eta)

                    for target_action in target_valid_actions:
                        # Avoid duplicate update of the already-observed pair.
                        if (np.array_equal(target_state, state)
                                and int(target_action) == int(action)):
                            continue

                        pseudo = _pseudo_transition(
                            env=env,
                            real_state=state,
                            real_action=action,
                            real_next_state=next_state,
                            target_state=target_state,
                            target_action=target_action,
                            solar_delta=solar_delta,
                            outcome=outcome,
                        )

                        if pseudo is None:
                            continue

                        pseudo_reward, pseudo_next_state = pseudo
                        q_learning_update(
                            Q, target_state, int(target_action), pseudo_reward,
                            pseudo_next_state, beta, alpha, env.eta
                        )

            state = next_state
            if terminated or truncated:
                break

        if ep % TEST_FREQ == 0:
            test_reward = evaluate_greedy_policy(env, Q)
            test_rewards.append((ep, test_reward))
            if ep % 500 == 0:
                print(f"  SK episode {ep:5d}/{Nepisodes}  eps={eps:.3f}  "
                      f"avg greedy return = {test_reward:.4f}")

    np.save("test_rewards_sk.npy", np.array(test_rewards, dtype=np.float64))
    print("Structural-knowledge Q-Learning complete. Test rewards saved to test_rewards_sk.npy")

    return extract_policy(Q, env)

env = SensorTransmissionEnv()

Nepisodes = 10000   # Number of episodes to train
alpha = 0.1         # Learning rate
beta = 0.98         # Discount factor

policy1 = QLearning(env, beta, Nepisodes, alpha)
policy2 = QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha)

np.save("policy1.npy", policy1)
np.save("policy2.npy", policy2)

env.close()
