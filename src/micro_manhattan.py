import numpy as np
import matplotlib.pyplot as plt

# --- Config "Micro Manhattan" Style Rigoureux ---
GRID_SIZE = 10

# Scénario
# D0: (0,5) -> (9,5)
driver_path = [np.array([x, 5]) for x in range(10)]
d_start, d_end = np.array([0, 5]), np.array([9, 5])

# P0: (0,6) -> (9,6) (Long trajet parallèle)
p0_start, p0_end = np.array([0, 6]), np.array([9, 6])

# P1: (0,4) -> (4,4) (Court trajet)
p1_start, p1_end = np.array([0, 4]), np.array([4, 4])

# P2: (5,4) -> (9,4) (Court trajet)
p2_start, p2_end = np.array([5, 4]), np.array([9, 4])

def plot_manhattan_line(ax, p1, p2, color, style='-', alpha=1.0, linewidth=1):
    # Trace une ligne en L (Manhattan) : X puis Y
    mid = np.array([p2[0], p1[1]])
    ax.plot([p1[0], mid[0]], [p1[1], mid[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)
    ax.plot([mid[0], p2[0]], [mid[1], p2[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)

def draw_base_scenario(ax, title):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 10); ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(range(11)); ax.set_yticks(range(11))
    
    # Trace Conducteur (Base)
    px = [p[0] for p in driver_path]
    py = [p[1] for p in driver_path]
    ax.plot(px, py, c='blue', alpha=0.1, linewidth=4, label='Trajet Conducteur')
    ax.scatter(d_start[0], d_start[1], c='blue', marker='s', s=100, edgecolors='black', label='D0 Start')
    ax.scatter(d_end[0], d_end[1], c='blue', marker='X', s=100, alpha=0.6, label='D0 End')

    # Trace Passagers (Base - Non servis par défaut)
    for (ps, pe, pid) in [(p0_start, p0_end, 'P0'), (p1_start, p1_end, 'P1'), (p2_start, p2_end, 'P2')]:
        ax.scatter(ps[0], ps[1], c='red', marker='o', s=80, edgecolors='black')
        ax.scatter(pe[0], pe[1], c='red', marker='^', s=80, alpha=0.6)
        ax.text(ps[0], ps[1]+0.3, f"{pid}s", fontsize=9, color='red', fontweight='bold')
        ax.text(pe[0], pe[1]+0.3, f"{pid}e", fontsize=9, color='red', alpha=0.8)
        # Ligne de désir grise
        plot_manhattan_line(ax, ps, pe, 'gray', '--', 0.2)

def draw_match(ax, p_start, p_end, pickup_idx, dropoff_idx):
    pickup_pt = driver_path[pickup_idx]
    dropoff_pt = driver_path[dropoff_idx]
    
    # Marche vers Pickup
    plot_manhattan_line(ax, p_start, pickup_pt, 'red', ':', 0.8, linewidth=1.5)
    
    # Trajet Voiture
    segment = driver_path[pickup_idx : dropoff_idx+1]
    seg_x = [pt[0] for pt in segment]
    seg_y = [pt[1] for pt in segment]
    ax.plot(seg_x, seg_y, c='green', linewidth=4, alpha=0.6, zorder=5)
    
    # Marche depuis Dropoff
    plot_manhattan_line(ax, dropoff_pt, p_end, 'red', ':', 0.8, linewidth=1.5)
    
    # Points de rendez-vous
    ax.scatter(pickup_pt[0], pickup_pt[1], c='green', marker='o', s=60, edgecolors='white', zorder=10)
    ax.scatter(dropoff_pt[0], dropoff_pt[1], c='green', marker='s', s=60, edgecolors='white', zorder=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- PLOT 1: GREEDY ---
draw_base_scenario(ax1, "1. Greedy : P0 choisi (Gain 100€)\nP1 et P2 bloqués")
# P0 est matché sur tout le trajet (indices 0 à 9)
# On suppose qu'il marche de (0,6) vers (0,5) et de (9,5) vers (9,6)
draw_match(ax1, p0_start, p0_end, 0, 9)

# --- PLOT 2: ILP ---
draw_base_scenario(ax2, "2. Optimal ILP : P1 et P2 choisis (Gain 120€)\nP0 ignoré pour le bien commun")
# P1 matché de 0 à 4
draw_match(ax2, p1_start, p1_end, 0, 4)
# P2 matché de 5 à 9
draw_match(ax2, p2_start, p2_end, 5, 9)

plt.tight_layout()
plt.savefig('micro_manhattan.png')
print("Graphique style 'Simulation' généré : micro_manhattan.png")