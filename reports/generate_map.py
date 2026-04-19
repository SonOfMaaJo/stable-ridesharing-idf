import osmnx as ox
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

# 1. Configuration
place_name = "Cergy, France"
dist = 3000
G = ox.graph_from_address(place_name, dist=dist, network_type='drive')
nodes_gdf, _ = ox.graph_to_gdfs(G)

# 2. Simulation avec contraintes spatiales (Placement intelligent)
random.seed(100)
n_passengers = 5
n_drivers = 3

all_points = []

def get_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_far_from_others(new_pt, existing_pts, threshold=0.004):
    for pt in existing_pts:
        if get_dist(new_pt, pt) < threshold:
            return False
    return True

def select_valid_pair(G, nodes_list, existing_points, min_od=0.015, min_space=0.005):
    max_tries = 500
    for _ in range(max_tries):
        o_node = random.choice(nodes_list)
        d_node = random.choice(nodes_list)
        o_pt = (G.nodes[o_node]['x'], G.nodes[o_node]['y'])
        d_pt = (G.nodes[d_node]['x'], G.nodes[d_node]['y'])
        if get_dist(o_pt, d_pt) < min_od:
            continue
        if is_far_from_others(o_pt, existing_points, min_space) and \
           is_far_from_others(d_pt, existing_points, min_space):
            return o_node, d_node, o_pt, d_pt
    return None, None, None, None

nodes_list = list(G.nodes())
agents_data = []
occupied_points = []

for i in range(n_drivers):
    o, d, opt, dpt = select_valid_pair(G, nodes_list, occupied_points, min_od=0.02, min_space=0.006)
    if o:
        agents_data.append({'agent_id': f'D{i+1}', 'type': 'conducteur', 'o': o, 'd': d})
        occupied_points.extend([opt, dpt])

for i in range(n_passengers):
    o, d, opt, dpt = select_valid_pair(G, nodes_list, occupied_points, min_od=0.012, min_space=0.005)
    if o:
        agents_data.append({'agent_id': f'P{i+1}', 'type': 'passager', 'o': o, 'd': d})
        occupied_points.extend([opt, dpt])

# Sauvegarde CSV
csv_rows = []
for a in agents_data:
    csv_rows.append({'agent_id': a['agent_id'], 'type': a['type'], 'orig_node': a['o'], 'dest_node': a['d']})
pd.DataFrame(csv_rows).to_csv('agents_instance.csv', index=False)

# 3. Visualisation (Amélioration des routes)
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# TRACÉ DES ROUTES : Couleur plus sombre (#CCCCCC) et plus épais (1.2)
_, edges_gdf = ox.graph_to_gdfs(G)
edges_gdf.plot(ax=ax, color='#CCCCCC', linewidth=1.2, zorder=1)

dark_red = '#8B0000'
blue = '#0000FF'

def add_label(ax, x, y, text, color, is_dest=False):
    offset = 0.0006 if is_dest else -0.0007
    ax.text(x, y + offset, text, fontsize=9, color=color, fontweight='bold',
            ha='center', va='center', zorder=25,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))

for a in agents_data:
    o_node, d_node = a['o'], a['d']
    color = dark_red if a['type'] == 'conducteur' else blue

    if a['type'] == 'conducteur':
        route = ox.shortest_path(G, o_node, d_node, weight='length')
        if route:
            x_r = [G.nodes[n]['x'] for n in route]
            y_r = [G.nodes[n]['y'] for n in route]
            ax.plot(x_r, y_r, color=color, linewidth=3, alpha=0.8, zorder=10)

    on = G.nodes[o_node]
    dn = G.nodes[d_node]
    ax.scatter(on['x'], on['y'], c=color, s=80, marker='o', edgecolors='white' if color==blue else 'black', zorder=20)
    add_label(ax, on['x'], on['y'], f"{a['agent_id']}s", color)
    ax.scatter(dn['x'], dn['y'], c=color, s=110, marker='^', edgecolors='white' if color==blue else 'black', zorder=20)
    add_label(ax, dn['x'], dn['y'], f"{a['agent_id']}a", color, is_dest=True)

ax.set_axis_off()
plt.title("Instance de Covoiturage", fontsize=16, fontweight='bold')
plt.savefig('map_covoiturage.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Graphique avec routes renforcées genere.")
