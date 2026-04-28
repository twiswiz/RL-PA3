import numpy as np

policy = np.load("policy2.npy")

plot1_slice = np.zeros((51, 51))
for t in range(51):
    for e in range(51):
        plot1_slice[t, e] = policy[t, 10, e, t]

counts_p1 = np.unique(plot1_slice, return_counts=True)
print("Plot 1 Distribution (Battery=10, max_seen=theta):")
for action, count in zip(counts_p1[0], counts_p1[1]):
    print(f"Action {int(action)}: {count} times ({count/2601*100:.1f}%)")

valid_states = []
for t in range(51):
    for m in range(t, 51):
        valid_states.append(policy[t, 10, 0, m])

counts_p2 = np.unique(valid_states, return_counts=True)
total_valid_p2 = len(valid_states)
print("\nPlot 2 Distribution (Battery=10, Estimate=0, max_seen >= theta):")
for action, count in zip(counts_p2[0], counts_p2[1]):
    print(f"Action {int(action)}: {count} times ({count/total_valid_p2*100:.1f}%)")
