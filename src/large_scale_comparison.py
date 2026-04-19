import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import time

# --- Configuration ---
GRID_SIZE = 40
NUM_DRIVERS = 70
CAR_CAPACITY = 4
SPEED_WALK = 5.0
SPEED_CAR = 30.0
UNIT_DIST_KM = 0.5
VALUE_OF_TIME = 15.0
WALKING_PENALTY = 1.5
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

class Agent:
    def __init__(self, id, type_agent, grid_size=GRID_SIZE):
        self.id = id
        self.type = type_agent
        self.start = np.random.randint(0, grid_size, size=2)
        self.end = np.random.randint(0, grid_size, size=2)
        self.path = []

    def generate_fixed_path(self):
        path = []
        current = self.start.copy()
        target = self.end
        if current[0] != target[0]:
            sx = 1 if target[0] > current[0] else -1
            for x in range(current[0], target[0] + sx, sx): path.append(np.array([x, current[1]]))
        corner = np.array([target[0], current[1]])
        if len(path) == 0 or not np.array_equal(path[-1], corner): path.append(corner)
        if current[1] != target[1]:
            sy = 1 if target[1] > current[1] else -1
            for y in range(current[1] + sy, target[1] + sy, sy): path.append(np.array([target[0], y]))
        self.path = path

def manhattan_dist(p1, p2): return np.sum(np.abs(p1 - p2))

def calculate_trip_cost_money(dist_walk, dist_ride):
    return ((dist_walk * UNIT_DIST_KM / SPEED_WALK) * WALKING_PENALTY + (dist_ride * UNIT_DIST_KM / SPEED_CAR)) * VALUE_OF_TIME

def get_potential_deals(drivers, passengers, compat_matrix_dp):
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
                    if cost < best_cost: best_cost = cost; best_idx = (ip, idrop)
            
            gain = calculate_trip_cost_money(manhattan_dist(p.start, p.end), 0) - best_cost
            if gain > 0: deals[(i, j)] = {'gain': gain, 'indices': best_idx}
    return deals

def solve_greedy(drivers, passengers, deals, compat_matrix_pp):
    # Tri par gain décroissant
    sorted_deals = sorted(deals.items(), key=lambda x: x[1]['gain'], reverse=True)
    
    car_occupants = {d.id: [] for d in drivers}
    matched_p = set()
    total_savings = 0
    
    for (d_idx, p_idx), info in sorted_deals:
        if p_idx in matched_p: continue
        if len(car_occupants[drivers[d_idx].id]) >= CAR_CAPACITY: continue
        
        # Vérif Clique
        current_pax = car_occupants[drivers[d_idx].id]
        if all(compat_matrix_pp[p_idx, existing] == 1 for existing in current_pax):
            matched_p.add(p_idx)
            car_occupants[drivers[d_idx].id].append(p_idx)
            total_savings += info['gain']
            
    return total_savings

def solve_ilp(drivers, passengers, deals, compat_matrix_pp):
    prob = LpProblem("Carpooling_Large", LpMaximize)
    x = {}
    
    for k in deals.keys(): x[k] = LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary)
    
    prob += lpSum([x[k] * deals[k]['gain'] for k in x])
    
    for j in range(len(passengers)): 
        prob += lpSum([x[(i, j)] for i in range(len(drivers)) if (i, j) in x]) <= 1
        
    for i in range(len(drivers)): 
        prob += lpSum([x[(i, j)] for j in range(len(passengers)) if (i, j) in x]) <= CAR_CAPACITY
        
    for i in range(len(drivers)):
        active = [j for j in range(len(passengers)) if (i, j) in x]
        # Optimisation : Loop réduite
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                p_a, p_b = active[idx_a], active[idx_b]
                if compat_matrix_pp[p_a, p_b] == 0:
                    prob += x[(i, p_a)] + x[(i, p_b)] <= 1
                    
    # Time limit augmenté pour les 70 drivers
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=90))
    
    total_savings = 0
    for k, v in x.items():
        if v.varValue == 1: total_savings += deals[k]['gain']
    return total_savings

# --- SIMULATION ---
drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
for d in drivers: d.generate_fixed_path()

# SCENARIO 1 : Évolution du Nb Passagers (Compat = 0.5)
pax_counts = [50, 100, 200, 300, 500]
res_s1_greedy = []
res_s1_ilp = []

print("--- Scénario 1 : Impact du Nombre de Passagers (Compat 50%) ---")
for n in pax_counts:
    np.random.seed(SEED + n)
    passengers = [Agent(i, 'P') for i in range(n)]
    
    # Génération Matrices
    c_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, n), p=[0.5, 0.5])
    c_pp = np.zeros((n, n))
    for i in range(n):
        c_pp[i, i] = 1
        for j in range(i+1, n):
            v = np.random.choice([0, 1], p=[0.5, 0.5])
            c_pp[i, j] = v; c_pp[j, i] = v
            
    deals = get_potential_deals(drivers, passengers, c_dp)
    
    start = time.time()
    g_save = solve_greedy(drivers, passengers, deals, c_pp)
    print(f"Pax {n}: Greedy terminé ({time.time()-start:.2f}s)")
    
    start = time.time()
    i_save = solve_ilp(drivers, passengers, deals, c_pp)
    print(f"Pax {n}: ILP terminé ({time.time()-start:.2f}s)")
    
    res_s1_greedy.append(g_save)
    res_s1_ilp.append(i_save)

# SCENARIO 2 : Évolution de la Compatibilité (Pax = 300)
rates = [0.2, 0.4, 0.6, 0.8, 1.0]
res_s2_greedy = []
res_s2_ilp = []
N_FIXED = 300

print("\n--- Scénario 2 : Impact de la Compatibilité (300 Pax) ---")
passengers = [Agent(i, 'P') for i in range(N_FIXED)] # Population fixe

for r in rates:
    np.random.seed(int(r*1000))
    
    c_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, N_FIXED), p=[1-r, r])
    c_pp = np.zeros((N_FIXED, N_FIXED))
    for i in range(N_FIXED):
        c_pp[i, i] = 1
        for j in range(i+1, N_FIXED):
            v = np.random.choice([0, 1], p=[1-r, r])
            c_pp[i, j] = v; c_pp[j, i] = v
            
    deals = get_potential_deals(drivers, passengers, c_dp)
    
    res_s2_greedy.append(solve_greedy(drivers, passengers, deals, c_pp))
    res_s2_ilp.append(solve_ilp(drivers, passengers, deals, c_pp))
    print(f"Rate {r}: Simulation terminée")

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1
ax1.set_title(f"Gain Social vs Nombre de Passagers\n({NUM_DRIVERS} Conducteurs, Compat 50%)")
ax1.set_xlabel("Nombre de Passagers")
ax1.set_ylabel("Gain Total (€)")
ax1.plot(pax_counts, res_s1_greedy, label='Greedy (Glouton)', marker='o', linestyle='--')
ax1.plot(pax_counts, res_s1_ilp, label='ILP (Optimal)', marker='s', linewidth=2)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2
ax2.set_title(f"Gain Social vs Compatibilité\n(300 Pax, {NUM_DRIVERS} Conducteurs)")
ax2.set_xlabel("Taux de Compatibilité")
ax2.set_ylabel("Gain Total (€)")
ax2.plot(rates, res_s2_greedy, label='Greedy (Glouton)', marker='o', linestyle='--')
ax2.plot(rates, res_s2_ilp, label='ILP (Optimal)', marker='s', linewidth=2)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('large_scale_comparison.png')
print("\nGraphique sauvegardé : large_scale_comparison.png")