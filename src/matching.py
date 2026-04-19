import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from src.agents import ALPHA_C, calculate_sdc
from tqdm import tqdm
from functools import lru_cache

class StableMatching:
    def __init__(self, passengers, drivers, graph, current_time):
        self.passengers = passengers
        self.drivers = drivers
        self.graph = graph
        self.current_time = current_time
        
        # Indexation spatiale
        self.nodes_data = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
        self.node_list = list(self.nodes_data.keys())
        self.node_coords = np.array([self.nodes_data[n] for n in self.node_list])
        self.kdtree = KDTree(self.node_coords)
        
        # Cache local pour ce pas de temps
        self._dist_cache = {}

    def get_dist(self, u, v):
        """Calcule la distance Dijkstra avec mise en cache."""
        pair = tuple(sorted((u, v)))
        if pair not in self._dist_cache:
            try:
                self._dist_cache[pair] = nx.shortest_path_length(self.graph, u, v, weight='length')
            except:
                self._dist_cache[pair] = float('inf')
        return self._dist_cache[pair]

    def build_preference_lists(self, search_radius=2000, top_k=5):
        passenger_prefs = {}
        driver_prefs = {d_id: [] for d_id in self.drivers}
        
        # Indexation des conducteurs par nœuds
        node_to_drivers = {}
        for d in self.drivers.values():
            for node in d.fixed_path:
                if node not in node_to_drivers: node_to_drivers[node] = []
                node_to_drivers[node].append(d.id)

        for p in tqdm(self.passengers, desc=f"Pref @ t={self.current_time}s", leave=False):
            options = []
            
            # 1. Distance de marche REELLE
            d_direct = self.get_dist(p.origin, p.destination)
            cost_m = p.get_walking_cost(d_direct, self.current_time)

            # 2. Pré-filtrage spatial (Euclidien)
            p_origin_coords = np.array(self.nodes_data[p.origin])
            indices = self.kdtree.query_ball_point(p_origin_coords, search_radius)
            nearby_nodes = set([self.node_list[i] for i in indices])
            
            candidate_ids = set()
            for n in nearby_nodes:
                if n in node_to_drivers: candidate_ids.update(node_to_drivers[n])

            # 3. Filtrage Top-K : On ne garde que les conducteurs dont un point est proche
            refined_candidates = []
            for d_id in candidate_ids:
                d = self.drivers[d_id]
                if not d.is_compatible(p.profile): continue
                
                # Trouver le nœud du trajet le plus proche géométriquement (très rapide)
                path_nodes_coords = np.array([self.nodes_data[n] for n in d.fixed_path])
                dists_sq = np.sum((path_nodes_coords - p_origin_coords)**2, axis=1)
                best_node_idx = np.argmin(dists_sq)
                best_node = d.fixed_path[best_node_idx]
                min_geom_dist = np.sqrt(dists_sq[best_node_idx])
                
                if min_geom_dist <= search_radius:
                    refined_candidates.append((d_id, best_node, min_geom_dist))
            
            # On trie par distance géométrique et on ne garde que les Top-K
            refined_candidates.sort(key=lambda x: x[2])
            refined_candidates = refined_candidates[:top_k]

            # 4. Calcul Dijkstra précis uniquement pour les Top-K
            for d_id, meet_node, _ in refined_candidates:
                d = self.drivers[d_id]
                real_pickup_dist = self.get_dist(meet_node, p.origin)
                
                if real_pickup_dist <= search_radius:
                    ride_dist = d_direct * 0.8 
                    t_arrival = self.current_time + (real_pickup_dist/1.39) + (ride_dist/10.0)
                    ride_cost = (ALPHA_C * (ride_dist/1000.0)) + calculate_sdc(t_arrival, p.t_star)
                    
                    if ride_cost < cost_m:
                        options.append((d_id, ride_cost))
                        driver_prefs[d_id].append((p.id, real_pickup_dist))

            options.sort(key=lambda x: x[1])
            passenger_prefs[p.id] = [x[0] for x in options] + [None]

        for d_id in driver_prefs:
            driver_prefs[d_id].sort(key=lambda x: x[1])
            driver_prefs[d_id] = [x[0] for x in driver_prefs[d_id]]

        return passenger_prefs, driver_prefs

    def solve(self, p_prefs, d_prefs):
        unmatched_p = [p.id for p in self.passengers]
        p_proposals_idx = {p.id: 0 for p in self.passengers}
        matches = {d_id: [] for d_id in self.drivers}
        
        while unmatched_p:
            p_id = unmatched_p.pop(0)
            if p_id not in p_prefs: continue
            p_l = p_prefs[p_id]
            if p_proposals_idx[p_id] >= len(p_l): continue
            
            target_d_id = p_l[p_proposals_idx[p_id]]
            p_proposals_idx[p_id] += 1
            if target_d_id is None: continue
            
            d_matches = matches[target_d_id]
            d_matches.append(p_id)
            d_list = d_prefs[target_d_id]
            d_matches.sort(key=lambda x: d_list.index(x) if x in d_list else 999)
            
            d = self.drivers[target_d_id]
            if len(d_matches) > (d.capacity - len(d.occupants)):
                rejected = d_matches.pop()
                unmatched_p.append(rejected)
        return matches
