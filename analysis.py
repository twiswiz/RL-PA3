import numpy as np
import matplotlib.pyplot as plt

def analyze_policies():
    # Load the Structural Knowledge policy
    try:
        policy = np.load("policy2.npy")
    except FileNotFoundError:
        print("Error: policy2.npy not found. Make sure you ran train.py first.")
        return
        
    # Shape is (51, 11, 51, 51) corresponding to:
    # [theta_idx, battery, estimate_idx, max_idx]

    # ---------------------------------------------------------
    # Deliverable 2: Pattern for Action 1 (Code action 0: Do Not Transmit)
    # ---------------------------------------------------------
    # Fix battery = 10 (fully charged), max_seen = theta (no pending high spike)
    # Plot heatmap of actions over (theta, estimate)
    heatmap_a1 = np.zeros((51, 51))
    for t in range(51):
        for e in range(51):
            heatmap_a1[t, e] = policy[t, 10, e, t]

    plt.figure(figsize=(8, 6))
    # 0: Purple (No Transmit), 1: Teal (Transmit True), 2: Yellow (Transmit Max)
    plt.imshow(heatmap_a1, origin='lower', cmap='viridis') 
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['0: No Transmit', '1: Transmit True', '2: Transmit Max'])
    plt.xlabel('Central Station Estimate (e_idx)')
    plt.ylabel('True Pollution (theta_idx)')
    plt.title('Optimal Action with Battery=10, max_seen = true pollution')
    
    # Save the plot for the report
    plt.savefig('action1_pattern.png', dpi=150)
    print("Saved action1_pattern.png. Notice the diagonal band showing 'No Transmit' when theta ≈ estimate.")
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 3: Pattern for Action 3 (Code action 2: Transmit Max)
    # ---------------------------------------------------------
    # Fix battery = 10, fix estimate = 0 (station assumes 0 pollution, risking high penalty)
    # Vary theta and max_seen
    heatmap_a3 = np.zeros((51, 51))
    for t in range(51):
        for m in range(t, 51): # max_seen must be logically >= theta
            heatmap_a3[t, m] = policy[t, 10, 0, m]

    # Mask mathematically invalid states where max_seen < theta
    masked_heatmap = np.ma.masked_where(np.tril(np.ones((51, 51)), -1), heatmap_a3)

    plt.figure(figsize=(8, 6))
    plt.imshow(masked_heatmap, origin='lower', cmap='viridis')
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['0: No Transmit', '1: Transmit True', '2: Transmit Max'])
    plt.xlabel('Max Seen Pollution (m_idx)')
    plt.ylabel('True Pollution (theta_idx)')
    plt.title('Optimal Action with Battery=10, Estimate=0')
    
    # Save the plot for the report
    plt.savefig('action3_pattern.png', dpi=150)
    print("Saved action3_pattern.png. Notice the region mapping to 'Transmit Max' when max_seen > theta.")
    plt.close()

if __name__ == "__main__":
    analyze_policies()
