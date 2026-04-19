import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# --- Configuration Grande Échelle ---
GRID_SIZE = 40
NUM_DRIVERS = 70
CAR_CAPACITY = 4
PASSENGER_COUNTS = [50, 100, 200, 300, 500]
COMPATIBILITY_PROFILES = [0.2, 0.5, 0.8]

SPEED_WALK = 5.0
SPEED_CAR = 30.0
UNIT_DIST_KM = 0.5
VALUE_OF_TIME = 15.0
WALKING_PENALTY = 1.5
SEED = 777

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

def solve_matching_ilp(drivers, passengers, deals, compat_matrix_pp):
    prob = LpProblem("Carpooling_Density", LpMaximize)
    x = {}
    for k in deals.keys(): x[k] = LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary)
    prob += lpSum([x[k] * deals[k]['gain'] for k in x])
    for j in range(len(passengers)): 
        prob += lpSum([x[(i, j)] for i in range(len(drivers)) if (i, j) in x]) <= 1
    for i in range(len(drivers)): 
        prob += lpSum([x[(i, j)] for j in range(len(passengers)) if (i, j) in x]) <= CAR_CAPACITY
    for i in range(len(drivers)):
        active = [j for j in range(len(passengers)) if (i, j) in x]
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                if compat_matrix_pp[active[idx_a], active[idx_b]] == 0:
                    prob += x[(i, active[idx_a])] + x[(i, active[idx_b])] <= 1
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    driver_occupancy = {d.id: 0 for d in drivers}
    matched_count = 0
    for k, v in x.items():
        if v.varValue == 1:
            driver_occupancy[drivers[k[0]].id] += 1
            matched_count += 1
    active_drivers = [occ for occ in driver_occupancy.values() if occ > 0]
    avg_occupancy = np.mean(active_drivers) if active_drivers else 0
    return avg_occupancy, matched_count

# --- MAIN LOOP ---
drivers = [Agent(i, 'D') for i in range(NUM_DRIVERS)]
for d in drivers: d.generate_fixed_path()

results = {rate: {'occupancy': [], 'matches': []} for rate in COMPATIBILITY_PROFILES}

print(f"Simulation Densité (70 Drivers): Test jusqu'à 500 Pax")

for n_pax in PASSENGER_COUNTS:
    print(f"--- Densité: {n_pax} Passagers ---")
    np.random.seed(SEED + n_pax)
    passengers = [Agent(i, 'P') for i in range(n_pax)]
    for rate in COMPATIBILITY_PROFILES:
        c_dp = np.random.choice([0, 1], size=(NUM_DRIVERS, n_pax), p=[1-rate, rate])
        c_pp = np.zeros((n_pax, n_pax))
        for i in range(n_pax):
            c_pp[i, i] = 1
            for j in range(i+1, n_pax):
                v = np.random.choice([0, 1], p=[1-rate, rate])
                c_pp[i, j] = v; c_pp[j, i] = v
        
        deals = get_potential_deals(drivers, passengers, c_dp)
        occ, matches = solve_matching_ilp(drivers, passengers, deals, c_pp)
        results[rate]['occupancy'].append(occ)
        results[rate]['matches'].append(matches)
        print(f"  Rate {rate}: Occup={occ:.2f}, Matches={matches}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = {0.2: 'tab:red', 0.5: 'tab:orange', 0.8: 'tab:green'}

ax1.set_title("Remplissage Moyen vs Densité (70 Drivers)")
ax1.set_xlabel("Nombre de Passagers")
ax1.set_ylabel("Passagers / Voiture Active")
ax1.grid(True, linestyle='--', alpha=0.5)
for rate in COMPATIBILITY_PROFILES:
    ax1.plot(PASSENGER_COUNTS, results[rate]['occupancy'], label=f'Compat {int(rate*100)}%', color=colors[rate], marker='o')
ax1.legend()

ax2.set_title("Volume de Passagers servis (Capacité Max: 280)")
ax2.set_xlabel("Nombre de Passagers")
ax2.set_ylabel("Nombre de Matchs")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axhline(y=NUM_DRIVERS*CAR_CAPACITY, color='black', linestyle=':', label='Capacité Système')
for rate in COMPATIBILITY_PROFILES:
    ax2.plot(PASSENGER_COUNTS, results[rate]['matches'], label=f'Compat {int(rate*100)}%', color=colors[rate], marker='s')
ax2.legend()

plt.tight_layout()
plt.savefig('density_impact.png')