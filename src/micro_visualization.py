import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres du scénario pédagogique ---
# Un conducteur D0 (Capacité 2)
# P0 : Trajet long, Gain = 100€
# P1 : Trajet court, Gain = 60€
# P2 : Trajet court, Gain = 60€
# 
# COMPATIBILITÉ :
# P0 est INCOMPATIBLE avec P1 et P2.
# P1 et P2 sont COMPATIBLES entre eux.

def plot_micro(ax, title, selected_pax, total_gain):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 11); ax.set_ylim(0, 10)
    ax.axis('off')

    # Dessin de la route (Ligne du conducteur)
    ax.plot([0, 10], [5, 5], color='navy', linewidth=10, alpha=0.1, zorder=1)
    ax.text(5, 5.5, "Trajet du Conducteur (Capacité: 2)", color='navy', ha='center', fontweight='bold')

    # Dessin des passagers potentiels
    # P0 (Rouge)
    p0_active = 'P0' in selected_pax
    ax.arrow(0.5, 6.5, 9, 0, width=0.2, color='red', alpha=1.0 if p0_active else 0.2, 
             length_includes_head=True, zorder=2)
    ax.text(5, 7, f"P0: Gain 100€ {'(CHOISI)' if p0_active else ''}", color='red', ha='center')

    # P1 (Vert)
    p1_active = 'P1' in selected_pax
    ax.arrow(0.5, 3.5, 4, 0, width=0.2, color='green', alpha=1.0 if p1_active else 0.2, 
             length_includes_head=True, zorder=2)
    ax.text(2.5, 3, f"P1: Gain 60€ {'(CHOISI)' if p1_active else ''}", color='green', ha='center')

    # P2 (Vert)
    p2_active = 'P2' in selected_pax
    ax.arrow(5.5, 3.5, 4, 0, width=0.2, color='green', alpha=1.0 if p2_active else 0.2, 
             length_includes_head=True, zorder=2)
    ax.text(7.5, 3, f"P2: Gain 60€ {'(CHOISI)' if p2_active else ''}", color='green', ha='center')

    # Cadre de résultat
    ax.text(5, 1, f"GAIN SOCIAL TOTAL: {total_gain} €", fontsize=14, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# LOGIQUE GREEDY : 
# 1. Il voit P0 (100€) comme le meilleur gain. Il le prend.
# 2. Il veut prendre P1, mais P0 et P1 sont incompatibles. Refusé.
# 3. Il veut prendre P2, mais P0 et P2 sont incompatibles. Refusé.
# Total = 100€
plot_micro(ax1, "1. Algorithme Glouton (Greedy)\nPriorité au profit individuel maximum", ['P0'], 100)

# LOGIQUE ILP (OPTIMAL) :
# 1. Il analyse toutes les combinaisons.
# 2. Groupe {P0} = 100€
# 3. Groupe {P1, P2} = 60 + 60 = 120€ (Possible car P1 et P2 sont compatibles)
# 4. Il choisit {P1, P2}
# Total = 120€
plot_micro(ax2, "2. Algorithme Optimal (ILP)\nMaximise l'intérêt de la collectivité", ['P1', 'P2'], 120)

plt.tight_layout()
plt.savefig('micro_visualization.png')
print("Visualisation microscopique générée : micro_visualization.png")
