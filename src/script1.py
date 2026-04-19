import numpy as np
import matplotlib.pyplot as plt
import random

# --- Configuration ---
GRID_SIZE = 20
NUM_DRIVERS = 5
NUM_PASSENGERS = 10
CAR_CAPACITY = 4 # Capacité max de passagers par voiture

# Vitesse et Coûts
SPEED_WALK = 5.0   # km/h
SPEED_CAR = 30.0   # km/h
UNIT_DIST_KM = 0.5
VALUE_OF_TIME = 15.0
WALKING_PENALTY = 1.5

# Taux de compatibilité
# On a besoin de haute compatibilité pour former des groupes > 1
COMPATIBILITY_RATE_D_P = 0.8 # Conducteur-Passager
COMPATIBILITY_RATE_P_P = 0.8 # Passager-Passager

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

class Agent:
    def __init__(self, id, type_agent):
        self.id = id
        self.type = type_agent
        self.start = np.random.randint(0, GRID_SIZE, size=2)
        self.end = np.random.randint(0, GRID_SIZE, size=2)
        self.path = []

    def generate_fixed_path(self):
        """Génère un chemin fixe en 'L' pour le conducteur."""
        path = []
        current = self.start.copy()
        target = self.end

        # 1. Déplacement en X
        step_x = 1 if target[0] > current[0] else -1
        if target[0] != current[0]:
            for x in range(current[0], target[0] + step_x, step_x):
                path.append(np.array([x, current[1]]))

        # 2. Déplacement en Y
        step_y = 1 if target[1] > current[1] else -1
        if target[1] != current[1]:
            start_y = current[1] + step_y if target[0] != current[0] else current[1] + step_y
            # Petit fix pour ne pas manquer le coin ou le doubler selon le cas
            # Simplification: on repart du coin
            start_y_corrected = current[1] + step_y
            if target[0] == current[0]: start_y_corrected = current[1] + step_y # Si pas de X
            else: start_y_corrected = current[1] + step_y

            # Re-calcul propre du L pour éviter les bugs d'indices
            path = []
            # Segment X
            if current[0] != target[0]:
                sx = 1 if target[0] > current[0] else -1
                for x in range(current[0], target[0] + sx, sx):
                    path.append(np.array([x, current[1]]))
            # Coin
            corner = np.array([target[0], current[1]])
            if len(path) == 0 or not np.array_equal(path[-1], corner):
                 path.append(corner)
            # Segment Y
            if current[1] != target[1]:
                sy = 1 if target[1] > current[1] else -1
                for y in range(current[1] + sy, target[1] + sy, sy):
                     path.append(np.array([target[0], y]))

        self.path = path
        return path

def manhattan_dist(p1, p2):
    return np.sum(np.abs(p1 - p2))

def calculate_trip_cost_money(dist_walk, dist_ride):
    dist_km_walk = dist_walk * UNIT_DIST_KM
    dist_km_ride = dist_ride * UNIT_DIST_KM
    time_walk_h = dist_km_walk / SPEED_WALK
    time_ride_h = dist_km_ride / SPEED_CAR
    total_cost = (time_walk_h * WALKING_PENALTY + time_ride_h) * VALUE_OF_TIME
    return total_cost

def solve_matching_multi(drivers, passengers, compat_matrix_dp, compat_matrix_pp):
    """
    Résout le problème 1-to-Many avec contraintes de clique.
    Approche Gloutonne (Greedy).
    """

    # 1. Lister toutes les "Offres" individuelles possibles
    # Une offre = (Conducteur, Passager, Gain, InfosTrajet)
    possible_deals = []

    for i, d in enumerate(drivers):
        for j, p in enumerate(passengers):

            # Check Compat Conducteur <-> Passager
            if compat_matrix_dp[i, j] == 0:
                continue

            # Trouver le meilleur trajet sur la ligne fixe
            best_trip_cost = np.inf
            best_indices = None

            dists_start_to_path = [manhattan_dist(p.start, pt) for pt in d.path]
            dists_path_to_end = [manhattan_dist(pt, p.end) for pt in d.path]
            path_len = len(d.path)

            for idx_p in range(path_len):
                walk_1 = dists_start_to_path[idx_p]
                for idx_d in range(idx_p, path_len):
                    walk_2 = dists_path_to_end[idx_d]
                    ride_dist = idx_d - idx_p
                    cost = calculate_trip_cost_money(walk_1 + walk_2, ride_dist)

                    if cost < best_trip_cost:
                        best_trip_cost = cost
                        best_indices = (idx_p, idx_d)

            # Calcul Gain
            walk_solo_dist = manhattan_dist(p.start, p.end)
            cost_solo = calculate_trip_cost_money(walk_solo_dist, 0)
            saving = cost_solo - best_trip_cost

            if saving > 0:
                possible_deals.append({
                    'driver_idx': i,
                    'passenger_idx': j,
                    'saving': saving,
                    'trip_indices': best_indices
                })

    # 2. Trier les deals par gain décroissant (On priorise les grosses économies)
    possible_deals.sort(key=lambda x: x['saving'], reverse=True)

    # 3. Allocation Gloutonne avec vérification de Clique
    final_matches = []
    total_savings = 0

    # État actuel des voitures: driver_idx -> list of passenger_indices
    car_occupants = {d.id: [] for d in drivers}
    matched_passengers = set()

    for deal in possible_deals:
        d_idx = deal['driver_idx']
        p_idx = deal['passenger_idx']

        # Si passager déjà casé, on passe
        if p_idx in matched_passengers:
            continue

        # Si voiture pleine, on passe
        current_passengers = car_occupants[drivers[d_idx].id]
        if len(current_passengers) >= CAR_CAPACITY:
            continue

        # VÉRIFICATION CLIQUE (Compatibilité Passager <-> Passager)
        is_compatible_with_all = True
        for existing_p_idx in current_passengers:
            # Attention: compat_matrix_pp est symétrique, mais on vérifie [p_new, p_old]
            if compat_matrix_pp[p_idx, existing_p_idx] == 0:
                is_compatible_with_all = False
                break

        if is_compatible_with_all:
            # On accepte le deal
            matched_passengers.add(p_idx)
            car_occupants[drivers[d_idx].id].append(p_idx)
            total_savings += deal['saving']

            final_matches.append({
                'driver_idx': d_idx,
                'passenger_idx': p_idx,
                'pickup_idx': deal['trip_indices'][0],
                'dropoff_idx': deal['trip_indices'][1],
                'saving': deal['saving']
            })

    return final_matches, total_savings

# --- Simulation ---

drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
passengers = [Agent(i, 'P') for i in range(NUM_PASSENGERS)]

for d in drivers:
    d.generate_fixed_path()

# 1. Matrice Compat Conducteur-Passager (N_D x N_P)
compat_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, NUM_PASSENGERS), p=[1-COMPATIBILITY_RATE_D_P, COMPATIBILITY_RATE_D_P])

# 2. Matrice Compat Passager-Passager (N_P x N_P) - Symétrique
compat_pp = np.zeros((NUM_PASSENGERS, NUM_PASSENGERS))
for i in range(NUM_PASSENGERS):
    compat_pp[i, i] = 1 # Compatible avec soi-même
    for j in range(i+1, NUM_PASSENGERS):
        val = np.random.choice([0, 1], p=[1-COMPATIBILITY_RATE_P_P, COMPATIBILITY_RATE_P_P])
        compat_pp[i, j] = val
        compat_pp[j, i] = val

matches, total_savings = solve_matching_multi(drivers, passengers, compat_dp, compat_pp)

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

def plot_manhattan_line(ax, p1, p2, color, style='-', alpha=1.0, linewidth=1):
    ax.plot([p1[0], p2[0]], [p1[1], p1[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)
    ax.plot([p2[0], p2[0]], [p1[1], p2[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)

def draw_agents_base(ax, drivers, passengers):
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.grid(True, linestyle='--', alpha=0.3)

    for d in drivers:
        px = [pt[0] for pt in d.path]
        py = [pt[1] for pt in d.path]
        ax.plot(px, py, c='blue', alpha=0.1, linewidth=3)
        ax.scatter(d.start[0], d.start[1], c='blue', marker='s', s=100, edgecolors='black')
        ax.scatter(d.end[0], d.end[1], c='blue', marker='X', s=100, alpha=0.6)
        ax.text(d.start[0]+0.3, d.start[1]+0.3, f"D{d.id}s", fontsize=9, color='blue', fontweight='bold')
        ax.text(d.end[0]+0.3, d.end[1]+0.3, f"D{d.id}e", fontsize=9, color='blue', alpha=0.8)

    for p in passengers:
        ax.scatter(p.start[0], p.start[1], c='red', marker='o', s=80, edgecolors='black')
        ax.scatter(p.end[0], p.end[1], c='red', marker='^', s=80, alpha=0.6)
        ax.text(p.start[0]+0.3, p.start[1]+0.3, f"P{p.id}s", fontsize=9, color='red', fontweight='bold')
        ax.text(p.end[0]+0.3, p.end[1]+0.3, f"P{p.id}e", fontsize=9, color='red', alpha=0.8)

# Graphique 1
axes[0].set_title("1. Situation Initiale")
draw_agents_base(axes[0], drivers, passengers)

# Graphique 2
axes[1].set_title(f"2. Solution (Multi-Passagers, Capacité {CAR_CAPACITY})\nÉconomie: {total_savings:.2f} €")
draw_agents_base(axes[1], drivers, passengers)

matched_ids = set()
driver_load = {d.id: 0 for d in drivers}

for m in matches:
    d = drivers[m['driver_idx']]
    p = passengers[m['passenger_idx']]
    matched_ids.add(p.id)
    driver_load[d.id] += 1
    
    pickup_pt = d.path[m['pickup_idx']]
    dropoff_pt = d.path[m['dropoff_idx']]
    
    # On trace les lignes vertes exactement sur la grille (sans décalage bizarre)
    
    plot_manhattan_line(axes[1], p.start, pickup_pt, color='red', style=':', alpha=0.8, linewidth=1.5)
    
    # Trajet Voiture
    segment = d.path[m['pickup_idx'] : m['dropoff_idx']+1]
    seg_x = [pt[0] for pt in segment]
    seg_y = [pt[1] for pt in segment]
    # On utilise une ligne un peu transparente pour que la superposition assombrisse le trait
    axes[1].plot(seg_x, seg_y, c='green', linewidth=4, alpha=0.5, zorder=5)
    
    plot_manhattan_line(axes[1], dropoff_pt, p.end, color='red', style=':', alpha=0.8, linewidth=1.5)
    axes[1].scatter(pickup_pt[0], pickup_pt[1], c='green', marker='o', s=80, edgecolors='white', zorder=10)
    axes[1].scatter(dropoff_pt[0], dropoff_pt[1], c='green', marker='s', s=80, edgecolors='white', zorder=10)
    # Non-Matchés
for p in passengers:
    if p.id not in matched_ids:
        plot_manhattan_line(axes[1], p.start, p.end, color='gray', style='--', alpha=0.5, linewidth=1)

# Stats de remplissage
stats_text = "Remplissage:\n" + "\n".join([f"D{d_id}: {count}/{CAR_CAPACITY}" for d_id, count in driver_load.items() if count > 0])
axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('ride_matching_simulation.png')
print(f"Simulation terminée. {len(matches)} trajets passagers.")
print(f"Économie collective : {total_savings:.2f} €")
