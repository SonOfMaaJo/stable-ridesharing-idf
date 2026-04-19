import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# --- Configuration Grande Échelle (Surcharge) ---
GRID_SIZE = 40
NUM_DRIVERS = 70       # Capacité max théorique = 280 places
NUM_PASSENGERS = 500   # Demande > Offre
CAR_CAPACITY = 4

SPEED_WALK = 5.0
SPEED_CAR = 30.0
UNIT_DIST_KM = 0.5
VALUE_OF_TIME = 15.0
WALKING_PENALTY = 1.5
SEED = 999

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

def calculate_time_minutes(dist_walk, dist_ride):
    dist_km_walk = dist_walk * UNIT_DIST_KM
    dist_km_ride = dist_ride * UNIT_DIST_KM
    time_walk_h = dist_km_walk / SPEED_WALK
    time_ride_h = dist_km_ride / SPEED_CAR
    return (time_walk_h + time_ride_h) * 60

def calculate_trip_cost_money(dist_walk, dist_ride):
    dist_km_walk = dist_walk * UNIT_DIST_KM
    dist_km_ride = dist_ride * UNIT_DIST_KM
    time_walk_h = dist_km_walk / SPEED_WALK
    time_ride_h = dist_km_ride / SPEED_CAR
    return (time_walk_h * WALKING_PENALTY + time_ride_h) * VALUE_OF_TIME

def get_potential_deals(drivers, passengers, compat_matrix_dp):
    deals = {}
    for i, d in enumerate(drivers):
        for j, p in enumerate(passengers):
            if compat_matrix_dp[i, j] == 0: continue
            
            best_cost = 1e9
            best_info = None
            dists_s = [manhattan_dist(p.start, pt) for pt in d.path]
            dists_e = [manhattan_dist(pt, p.end) for pt in d.path]
            
            for ip in range(len(d.path)):
                for idrop in range(ip, len(d.path)):
                    walk_dist = dists_s[ip] + dists_e[idrop]
                    ride_dist = idrop - ip
                    cost = calculate_trip_cost_money(walk_dist, ride_dist)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_info = {'indices': (ip, idrop), 'walk_dist': walk_dist, 'ride_dist': ride_dist}
            
            solo_dist = manhattan_dist(p.start, p.end)
            gain = calculate_trip_cost_money(solo_dist, 0) - best_cost
            if gain > 0:
                deals[(i, j)] = {'gain': gain, 'info': best_info}
    return deals

def solve_matching_ilp(drivers, passengers, deals, compat_matrix_pp):
    prob = LpProblem("Carpooling_Time", LpMaximize)
    x = {}
    for k in deals.keys(): x[k] = LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary)
    
    prob += lpSum([x[k] * deals[k]['gain'] for k in x])
    
    for j in range(len(passengers)): 
        prob += lpSum([x[(i, j)] for i in range(len(drivers)) if (i, j) in x]) <= 1
        
    for i in range(len(drivers)): 
        prob += lpSum([x[(i, j)] for j in range(len(passengers)) if (i, j) in x]) <= CAR_CAPACITY
        
    for i in range(len(drivers)):
        active = [j for j in range(len(passengers)) if (i, j) in x]
        # Optimisation Loop
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                if compat_matrix_pp[active[idx_a], active[idx_b]] == 0:
                    prob += x[(i, active[idx_a])] + x[(i, active[idx_b])] <= 1
                    
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    
    matches = {}
    for k, v in x.items():
        if v.varValue == 1:
            matches[k[1]] = k[0]
    return matches

# --- MAIN LOOP ---
drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
passengers = [Agent(i, 'P') for i in range(NUM_PASSENGERS)]
for d in drivers: d.generate_fixed_path()

# Baseline
base_times = []
for p in passengers:
    dist = manhattan_dist(p.start, p.end)
    base_times.append(calculate_time_minutes(dist, 0))
avg_base_time = np.mean(base_times)

rates = np.linspace(0.1, 1.0, 10)
avg_times_global = []
avg_times_matched = []
match_percentages = []

print(f"Base Average Walking Time: {avg_base_time:.1f} min")

for rate in rates:
    np.random.seed(int(rate*1000))
    c_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, NUM_PASSENGERS), p=[1-rate, rate])
    c_pp = np.zeros((NUM_PASSENGERS, NUM_PASSENGERS))
    for i in range(NUM_PASSENGERS):
        c_pp[i, i] = 1
        for j in range(i+1, NUM_PASSENGERS):
            v = np.random.choice([0, 1], p=[1-rate, rate])
            c_pp[i, j] = v; c_pp[j, i] = v
            
    deals = get_potential_deals(drivers, passengers, c_dp)
    matches = solve_matching_ilp(drivers, passengers, deals, c_pp)
    
    current_times = []
    matched_times_only = []
    
    for p_idx, p in enumerate(passengers):
        if p_idx in matches:
            d_idx = matches[p_idx]
            deal_info = deals[(d_idx, p_idx)]['info']
            t = calculate_time_minutes(deal_info['walk_dist'], deal_info['ride_dist'])
            current_times.append(t)
            matched_times_only.append(t)
        else:
            t = calculate_time_minutes(manhattan_dist(p.start, p.end), 0)
            current_times.append(t)
            
    avg_times_global.append(np.mean(current_times))
    avg_times_matched.append(np.mean(matched_times_only) if matched_times_only else np.nan)
    match_percentages.append(len(matches) / NUM_PASSENGERS * 100)
    
    print(f"Rate {rate:.1f}: Global={np.mean(current_times):.1f} min, Matched={np.mean(matched_times_only):.1f} min, Service={len(matches)}/{NUM_PASSENGERS}")

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Taux de Compatibilité', fontsize=12)
ax1.set_ylabel('Temps de Trajet Moyen (min)', color=color, fontsize=12)

# Baseline
ax1.axhline(y=avg_base_time, color='gray', linestyle='--', label='Temps Marche Seule (Base)')

# Global Avg
ax1.plot(rates, avg_times_global, color=color, marker='o', linewidth=3, label='Moyenne Globale (Tous passagers)')

# Matched Avg (Les Chanceux)
ax1.plot(rates, avg_times_matched, color='orange', marker='x', linestyle=':', label='Moyenne (Covoitureurs Uniquement)')

ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Axe secondaire pour le % de matchs
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('% Passagers avec Covoiturage', color=color2, fontsize=12)
ax2.plot(rates, match_percentages, color=color2, alpha=0.3, linewidth=0, marker='s', label='% Matchs')
ax2.fill_between(rates, 0, match_percentages, color=color2, alpha=0.1)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f"Impact du Covoiturage sur le Temps ({NUM_PASSENGERS} Pax / {NUM_DRIVERS} Drivers)", fontsize=14)
plt.tight_layout()
plt.savefig('time_impact_analysis.png')
