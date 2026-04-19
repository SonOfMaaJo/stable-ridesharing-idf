import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# --- Configuration ---
GRID_SIZE = 20
NUM_DRIVERS = 5
NUM_PASSENGERS = 12
CAR_CAPACITY = 3 

SPEED_WALK = 5.0   
SPEED_CAR = 30.0   
UNIT_DIST_KM = 0.5 
VALUE_OF_TIME = 15.0 
WALKING_PENALTY = 1.5 

COMPATIBILITY_RATE_D_P = 0.7 
COMPATIBILITY_RATE_P_P = 0.7 

SEED = 555
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
        path = []
        current = self.start.copy()
        target = self.end
        if current[0] != target[0]:
            sx = 1 if target[0] > current[0] else -1
            for x in range(current[0], target[0] + sx, sx):
                path.append(np.array([x, current[1]]))
        corner = np.array([target[0], current[1]])
        if len(path) == 0 or not np.array_equal(path[-1], corner):
             path.append(corner)
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
    return (time_walk_h * WALKING_PENALTY + time_ride_h) * VALUE_OF_TIME

def get_potential_gains(drivers, passengers, compat_matrix_dp):
    """Calcule tous les gains individuels possibles."""
    deals = {}
    for i, d in enumerate(drivers):
        for j, p in enumerate(passengers):
            if compat_matrix_dp[i, j] == 0: continue
            
            best_cost = 1e9
            best_idx = None
            dists_s = [manhattan_dist(p.start, pt) for pt in d.path]
            dists_e = [manhattan_dist(pt, p.end) for pt in d.path]
            
            for ip in range(len(d.path)):
                for idrop in range(ip, len(d.path)):
                    cost = calculate_trip_cost_money(dists_s[ip] + dists_e[idrop], idrop - ip)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = (ip, idrop)
            
            gain = calculate_trip_cost_money(manhattan_dist(p.start, p.end), 0) - best_cost
            if gain > 0:
                deals[(i, j)] = {'gain': gain, 'indices': best_idx}
    return deals

def solve_matching_greedy(drivers, passengers, deals, compat_matrix_pp):
    """Approche Gloutonne."""
    sorted_deals = sorted(deals.items(), key=lambda x: x[1]['gain'], reverse=True)
    car_occupants = {d.id: [] for d in drivers}
    matched_p = set()
    final_matches = []
    total_savings = 0
    
    for (d_idx, p_idx), info in sorted_deals:
        if p_idx in matched_p: continue
        if len(car_occupants[drivers[d_idx].id]) >= CAR_CAPACITY: continue
        
        if all(compat_matrix_pp[p_idx, existing] == 1 for existing in car_occupants[drivers[d_idx].id]):
            matched_p.add(p_idx)
            car_occupants[drivers[d_idx].id].append(p_idx)
            total_savings += info['gain']
            final_matches.append({'driver_idx': d_idx, 'passenger_idx': p_idx, 
                                  'pickup_idx': info['indices'][0], 'dropoff_idx': info['indices'][1], 'saving': info['gain']})
    return final_matches, total_savings

def solve_matching_ilp(drivers, passengers, deals, compat_matrix_pp):
    """Approche Optimisée via Programmation Linéaire (PuLP)."""
    prob = LpProblem("Carpooling_Optimization", LpMaximize)
    
    # Variables de décision x[d, p]
    x = {}
    for (d_idx, p_idx) in deals.keys():
        x[(d_idx, p_idx)] = LpVariable(f"x_{d_idx}_{p_idx}", cat=LpBinary)
    
    # Objectif : Maximiser le gain total
    prob += lpSum([x[key] * deals[key]['gain'] for key in x])
    
    # Contrainte 1 : Un passager max 1 voiture
    for j in range(len(passengers)):
        prob += lpSum([x[(i, j)] for i in range(len(drivers)) if (i, j) in x]) <= 1
        
    # Contrainte 2 : Capacité des voitures
    for i in range(len(drivers)):
        prob += lpSum([x[(i, j)] for j in range(len(passengers)) if (i, j) in x]) <= CAR_CAPACITY
        
    # Contrainte 3 : Incompatibilité Passager-Passager (Clique)
    # Pour chaque conducteur, si Pi et Pj sont incompatibles, ils ne peuvent pas être ensemble
    for i in range(len(drivers)):
        active_p = [j for j in range(len(passengers)) if (i, j) in x]
        for idx_a in range(len(active_p)):
            for idx_b in range(idx_a + 1, len(active_p)):
                p_a = active_p[idx_a]
                p_b = active_p[idx_b]
                if compat_matrix_pp[p_a, p_b] == 0:
                    prob += x[(i, p_a)] + x[(i, p_b)] <= 1
                    
    # Résolution
    prob.solve(PULP_CBC_CMD(msg=0))
    
    final_matches = []
    total_savings = 0
    for (d_idx, p_idx), var in x.items():
        if var.varValue == 1:
            total_savings += deals[(d_idx, p_idx)]['gain']
            final_matches.append({'driver_idx': d_idx, 'passenger_idx': p_idx, 
                                  'pickup_idx': deals[(d_idx, p_idx)]['indices'][0], 
                                  'dropoff_idx': deals[(d_idx, p_idx)]['indices'][1], 
                                  'saving': deals[(d_idx, p_idx)]['gain']})
    return final_matches, total_savings

# --- Simulation ---
drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
passengers = [Agent(i, 'P') for i in range(NUM_PASSENGERS)]
for d in drivers: d.generate_fixed_path()

compat_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, NUM_PASSENGERS), p=[1-COMPATIBILITY_RATE_D_P, COMPATIBILITY_RATE_D_P])
compat_pp = np.zeros((NUM_PASSENGERS, NUM_PASSENGERS))
for i in range(NUM_PASSENGERS):
    compat_pp[i, i] = 1
    for j in range(i+1, NUM_PASSENGERS):
        v = np.random.choice([0, 1], p=[1-COMPATIBILITY_RATE_P_P, COMPATIBILITY_RATE_P_P])
        compat_pp[i, j] = v; compat_pp[j, i] = v

deals = get_potential_gains(drivers, passengers, compat_dp)

# Comparaison des méthodes
matches_g, savings_g = solve_matching_greedy(drivers, passengers, deals, compat_pp)
matches_i, savings_i = solve_matching_ilp(drivers, passengers, deals, compat_pp)

# --- Visualization (on utilise la solution ILP) ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

def plot_manhattan_line(ax, p1, p2, color, style='-', alpha=1.0, linewidth=1):
    ax.plot([p1[0], p2[0]], [p1[1], p1[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)
    ax.plot([p2[0], p2[0]], [p1[1], p2[1]], c=color, linestyle=style, alpha=alpha, linewidth=linewidth)

def draw_base(ax, drivers, passengers, title):
    ax.set_title(title)
    ax.set_xlim(-1, GRID_SIZE); ax.set_ylim(-1, GRID_SIZE); ax.grid(True, alpha=0.3)
    for d in drivers:
        ax.plot([pt[0] for pt in d.path], [pt[1] for pt in d.path], c='blue', alpha=0.1, linewidth=3)
        ax.scatter(d.start[0], d.start[1], c='blue', marker='s', s=100); ax.scatter(d.end[0], d.end[1], c='blue', marker='X', s=100, alpha=0.6)
        ax.text(d.start[0]+0.3, d.start[1]+0.3, f"D{d.id}s", fontsize=9, color='blue', fontweight='bold')
        ax.text(d.end[0]+0.3, d.end[1]+0.3, f"D{d.id}e", fontsize=9, color='blue', alpha=0.8)
    for p in passengers:
        ax.scatter(p.start[0], p.start[1], c='red', marker='o', s=80); ax.scatter(p.end[0], p.end[1], c='red', marker='^', s=80, alpha=0.6)
        ax.text(p.start[0]+0.3, p.start[1]+0.3, f"P{p.id}s", fontsize=9, color='red', fontweight='bold')
        ax.text(p.end[0]+0.3, p.end[1]+0.3, f"P{p.id}e", fontsize=9, color='red', alpha=0.8)

draw_base(axes[0], drivers, passengers, f"1. Greedy Method (Gain: {savings_g:.2f} €)")
draw_base(axes[1], drivers, passengers, f"2. ILP Optimal Method (Gain: {savings_i:.2f} €)")

# Dessiner Greedy
m_p_g = set()
for m in matches_g:
    d, p = drivers[m['driver_idx']], passengers[m['passenger_idx']]
    m_p_g.add(p.id)
    plot_manhattan_line(axes[0], p.start, d.path[m['pickup_idx']], 'red', ':', 0.8)
    axes[0].plot([pt[0] for pt in d.path[m['pickup_idx']:m['dropoff_idx']+1]], [pt[1] for pt in d.path[m['pickup_idx']:m['dropoff_idx']+1]], c='green', lw=3, alpha=0.5)
    plot_manhattan_line(axes[0], d.path[m['dropoff_idx']], p.end, 'red', ':', 0.8)

# Dessiner ILP
m_p_i = set()
for m in matches_i:
    d, p = drivers[m['driver_idx']], passengers[m['passenger_idx']]
    m_p_i.add(p.id)
    plot_manhattan_line(axes[1], p.start, d.path[m['pickup_idx']], 'red', ':', 0.8)
    axes[1].plot([pt[0] for pt in d.path[m['pickup_idx']:m['dropoff_idx']+1]], [pt[1] for pt in d.path[m['pickup_idx']:m['dropoff_idx']+1]], c='green', lw=3, alpha=0.5)
    plot_manhattan_line(axes[1], d.path[m['dropoff_idx']], p.end, 'red', ':', 0.8)

for p in passengers:
    if p.id not in m_p_g: plot_manhattan_line(axes[0], p.start, p.end, 'gray', '--', 0.3)
    if p.id not in m_p_i: plot_manhattan_line(axes[1], p.start, p.end, 'gray', '--', 0.3)

plt.tight_layout()
plt.savefig('ride_matching_simulation.py.png')
print(f"Greedy Savings: {savings_g:.2f} € | Matches: {len(matches_g)}")
print(f"ILP Savings   : {savings_i:.2f} € | Matches: {len(matches_i)}")
print(f"Amélioration ILP vs Greedy : {((savings_i - savings_g)/max(1,savings_g))*100:.1f} %")