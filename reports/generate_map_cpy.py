import osmnx as ox
import matplotlib.pyplot as plt
import random
import pandas as pd
import geopandas as gpd

# 1. Configuration et Téléchargement
place_name = "Cergy, France"
dist = 1500
print(f"Telechargement du reseau pour {place_name}...")
G = ox.graph_from_address(place_name, dist=dist, network_type='drive')

# Conversion en GeoDataFrames pour un tracé robuste
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

# 2. Simulation des Agents (Identique pour la DB)
random.seed(42)
n_passengers = 5
n_drivers = 3

all_nodes_list = list(G.nodes())
sample_nodes = random.sample(all_nodes_list, (n_passengers + n_drivers) * 2)

p_origs = sample_nodes[:n_passengers]
p_dests = sample_nodes[n_passengers : 2*n_passengers]
d_origs = sample_nodes[2*n_passengers : 2*n_passengers + n_drivers]
d_dests = sample_nodes[2*n_passengers + n_drivers:]

# 3. Sauvegarde de la DB
data = []
for i in range(n_passengers):
    data.append({'agent_id': f'P{i+1}', 'type': 'passager', 'orig_node': p_origs[i], 'dest_node': p_dests[i]})
for i in range(n_drivers):
    data.append({'agent_id': f'D{i+1}', 'type': 'conducteur', 'orig_node': d_origs[i], 'dest_node': d_dests[i]})
pd.DataFrame(data).to_csv('agents_instance.csv', index=False)

# 4. Visualisation via Matplotlib pur
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Tracer les routes en fond (Gris)
edges_gdf.plot(ax=ax, color='#CCCCCC', linewidth=0.8, zorder=1)

# Couleurs et styles
dark_red = '#8B0000'
blue = '#0000FF'

# Points de repere
landmarks = {
    "Gare Cergy-Prefecture": (2.036, 49.036),
    "CC Les 3 Fontaines": (2.034, 49.039),
    "CY Universite": (2.042, 49.038)
}
for name, (lon, lat) in landmarks.items():
    ax.scatter(lon, lat, c='green', s=100, marker='+', zorder=5)
    ax.text(lon, lat + 0.0003, name, fontsize=9, color='green', fontweight='bold', ha='center', zorder=6)

# Tracer les conducteurs
for i in range(n_drivers):
    route = ox.shortest_path(G, d_origs[i], d_dests[i], weight='length')
    if route:
        # Extraire les coordonnées
        x = [G.nodes[n]['x'] for n in route]
        y = [G.nodes[n]['y'] for n in route]
        ax.plot(x, y, color=dark_red, linewidth=3, zorder=10)
        
        # Ds et Da
        ax.scatter(x[0], y[0], c=dark_red, s=120, marker='o', edgecolors='black', zorder=15)
        ax.text(x[0], y[0], f' Ds_{i+1}', fontsize=10, color=dark_red, fontweight='bold', zorder=16)
        ax.scatter(x[-1], y[-1], c=dark_red, s=150, marker='^', edgecolors='black', zorder=15)
        ax.text(x[-1], y[-1], f' Da_{i+1}', fontsize=10, color=dark_red, fontweight='bold', zorder=16)

# Tracer les passagers
for i in range(n_passengers):
    o = G.nodes[p_origs[i]]
    d = G.nodes[p_dests[i]]
    # Ps
    ax.scatter(o['x'], o['y'], c=blue, s=100, marker='o', edgecolors='white', zorder=15)
    ax.text(o['x'], o['y'], f' Ps_{i+1}', fontsize=10, color=blue, fontweight='bold', zorder=16)
    # Pa
    ax.scatter(d['x'], d['y'], c=blue, s=130, marker='^', edgecolors='white', zorder=15)
    ax.text(d['x'], d['y'], f' Pa_{i+1}', fontsize=10, color=blue, fontweight='bold', zorder=16)

# Ajuster les limites pour cadrer tous les points
ax.set_xlim(nodes_gdf['x'].min() - 0.002, nodes_gdf['x'].max() + 0.002)
ax.set_ylim(nodes_gdf['y'].min() - 0.002, nodes_gdf['y'].max() + 0.002)

ax.set_axis_off()
plt.title("Instance de Covoiturage - Cergy Grand Centre", fontsize=16, fontweight='bold')

# Sauvegarde
plt.savefig('map_covoiturage.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Graphique genere avec GeoPandas (Succès).")
