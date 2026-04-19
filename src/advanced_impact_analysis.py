import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# --- Configuration Homogène Grande Échelle ---
GRID_SIZE = 40
NUM_DRIVERS = 70
NUM_PASSENGERS = 500 
CAPACITIES_TO_TEST = [2, 3, 5]

SPEED_WALK = 5.0
SPEED_CAR = 30.0
UNIT_DIST_KM = 0.5
VALUE_OF_TIME = 15.0
WALKING_PENALTY = 1.5
SEED = 2024

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

def get_potential_gains(drivers, passengers, compat_matrix_dp):
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

def solve_matching_ilp(drivers, passengers, deals, compat_matrix_pp, capacity):
    prob = LpProblem("Carpooling", LpMaximize)
    x = {}
    for k in deals.keys(): x[k] = LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary)
    
    prob += lpSum([x[k] * deals[k]['gain'] for k in x])
    for j in range(len(passengers)): 
        prob += lpSum([x[(i, j)] for i in range(len(drivers)) if (i, j) in x]) <= 1
    for i in range(len(drivers)): 
        prob += lpSum([x[(i, j)] for j in range(len(passengers)) if (i, j) in x]) <= capacity
    for i in range(len(drivers)):
        active = [j for j in range(len(passengers)) if (i, j) in x]
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                if compat_matrix_pp[active[idx_a], active[idx_b]] == 0:
                    prob += x[(i, active[idx_a])] + x[(i, active[idx_b])] <= 1
                    
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    
    matches_per_driver = {d.id: 0 for d in drivers}
    matched_passengers_count = 0
    for k, v in x.items():
        if v.varValue == 1:
            matched_passengers_count += 1
            matches_per_driver[drivers[k[0]].id] += 1
            
    active_drivers = [count for count in matches_per_driver.values() if count > 0]
    avg_occupancy = np.mean(active_drivers) if active_drivers else 0
    return matched_passengers_count, avg_occupancy

# --- MAIN LOOP ---
drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
passengers = [Agent(i, 'P') for i in range(NUM_PASSENGERS)]
for d in drivers: d.generate_fixed_path()

rates = np.linspace(0.1, 1.0, 10)
results = {cap: {'occupancy': [], 'service_rate': []} for cap in CAPACITIES_TO_TEST}

print(f"Analyse Capacité (500 Pax): {NUM_DRIVERS} Drivers")

for cap in CAPACITIES_TO_TEST:
    print(f"--- Capacité {cap} ---")
    for rate in rates:
        local_seed = int(rate * 1000)
        np.random.seed(local_seed)
        c_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, NUM_PASSENGERS), p=[1-rate, rate])
        c_pp = np.zeros((NUM_PASSENGERS, NUM_PASSENGERS))
        for i in range(NUM_PASSENGERS):
            c_pp[i, i] = 1
            for j in range(i+1, NUM_PASSENGERS):
                v = np.random.choice([0, 1], p=[1-rate, rate])
                c_pp[i, j] = v; c_pp[j, i] = v
        
        deals = get_potential_gains(drivers, passengers, c_dp)
        matches, occupancy = solve_matching_ilp(drivers, passengers, deals, c_pp, cap)
        
        results[cap]['occupancy'].append(occupancy)
        results[cap]['service_rate'].append((matches / NUM_PASSENGERS) * 100)
        print(f"Rate {rate:.1f}: Matchs={matches}, Occup={occupancy:.2f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = {2: 'tab:blue', 3: 'tab:orange', 5: 'tab:green'}
markers = {2: 'o', 3: 's', 5: '^'}

ax1.set_title(f"Taux de Service (Demande: {NUM_PASSENGERS} Pax)")
ax1.set_xlabel("Taux de Compatibilité")
ax1.set_ylabel("% Passagers servis")
ax1.grid(True, linestyle='--', alpha=0.5)
for cap in CAPACITIES_TO_TEST:
    ax1.plot(rates, results[cap]['service_rate'], label=f'Capacité {cap}', color=colors[cap], marker=markers[cap])
ax1.legend()

ax2.set_title(f"Remplissage Moyen ({NUM_DRIVERS} Conducteurs)")
ax2.set_xlabel("Taux de Compatibilité")
ax2.set_ylabel("Passagers / Voiture Active")
ax2.grid(True, linestyle='--', alpha=0.5)
for cap in CAPACITIES_TO_TEST:
    ax2.plot(rates, results[cap]['occupancy'], label=f'Capacité {cap}', color=colors[cap], marker=markers[cap])
ax2.legend()

plt.tight_layout()
plt.savefig('advanced_impact_analysis.png')
