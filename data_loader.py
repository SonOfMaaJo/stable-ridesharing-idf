import pandas as pd
import numpy as np
import os
from shapely import wkb
from scipy.spatial import KDTree

class DataLoader:
    """Gestionnaire de données pour la simulation IDF (v2 avec Géométries)."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.communes_df = None
        self.travel_time_matrix = {}
        self.insee_to_coords = {} # ID INSEE -> (x, y)
        self.kdtree = None
        self.insee_list = []

    def load_all(self, target_departments=None, light_mode=False):
        """
        Charge les données. 
        Si light_mode=True, ne charge que les coordonnées (utile pour la visu).
        """
        print("Chargement des données IDF en cours...")
        
        # 1. Médoïdes des communes (ID INSEE -> Géométrie WKB)
        full_communes = pd.read_parquet(os.path.join(self.data_dir, "communes_medoids.parquet"))
        
        if target_departments:
            print(f" -> Mode ZONE RESTREINTE : {target_departments}")
            self.communes_df = full_communes[full_communes['insee_id'].astype(str).str[:2].isin(target_departments)].copy()
        else:
            print(" -> Mode RÉSEAU ENTIER (IDF)")
            self.communes_df = full_communes.copy()

        valid_insee_set = set(self.communes_df['insee_id'])
        
        # Extraction des coordonnées X, Y
        coords_list = []
        self.insee_to_coords = {}
        self.insee_list = []
        for idx, row in self.communes_df.iterrows():
            point = wkb.loads(row['geometry'])
            x, y = point.x, point.y
            self.insee_to_coords[row['insee_id']] = (x, y)
            coords_list.append([x, y])
            self.insee_list.append(row['insee_id'])
            
        # Construction du KDTree
        self.kdtree = KDTree(np.array(coords_list))
        print(f" -> {len(self.insee_list)} communes indexées spatialement.")

        if light_mode:
            print(" -> Mode LIGHT : Graphe et OD Matrix ignorés.")
            return

        # 2. Matrice OD INSEE
        full_od = pd.read_parquet(os.path.join(self.data_dir, "od_matrix_insee.parquet"))
        if target_departments:
            self.od_matrix_df = full_od[
                (full_od['insee_origin'].isin(valid_insee_set)) & 
                (full_od['insee_destination'].isin(valid_insee_set))
            ].copy()
        else:
            self.od_matrix_df = full_od.copy()
        
        # 3. Temps de trajet et Graphe de communes
        tt_df = pd.read_parquet(os.path.join(self.data_dir, "communes_free_flow_travel_times.parquet"))
        
        # On ne garde que les temps de trajet utiles pour la zone
        if target_departments:
            tt_df = tt_df[
                (tt_df['origin'].isin(valid_insee_set)) & 
                (tt_df['destination'].isin(valid_insee_set))
            ].copy()

        self.travel_time_matrix = tt_df.set_index(['origin', 'destination'])['free_flow_travel_time'].to_dict()
        
        import networkx as nx
        self.commune_graph = nx.Graph()
        # On n'ajoute que les trajets courts (< 15 min) pour le graphe
        subset = tt_df[tt_df['free_flow_travel_time'] < 900] 
        for _, row in subset.iterrows():
            self.commune_graph.add_edge(row['origin'], row['destination'], weight=row['free_flow_travel_time'])
            
        print(f"Chargement terminé. Graphe de communes : {self.commune_graph.number_of_nodes()} nœuds.")

    def get_driver_path(self, origin_insee, dest_insee):
        """Calcule la séquence de communes traversées (Shortest Path)."""
        import networkx as nx
        try:
            return nx.shortest_path(self.commune_graph, origin_insee, dest_insee, weight='weight')
        except:
            return [origin_insee, dest_insee] # Fallback si pas de chemin trouvé

    def get_nearby_communes(self, insee_id, radius_km):
        """Trouve les communes voisines dans un rayon de X km (Section 5.1)."""
        if insee_id not in self.insee_to_coords:
            return [insee_id]
            
        center_coords = self.insee_to_coords[insee_id]
        # Conversion km -> degrés approx (1 deg ~ 111 km) ou mètres (selon le CRS)
        # Supposons que vos coordonnées sont en mètres ( Lambert 93 typique IDF)
        radius_m = radius_km * 1000.0
        
        indices = self.kdtree.query_ball_point(center_coords, radius_m)
        return [self.insee_list[i] for i in indices]

    def get_travel_time(self, origin_insee, dest_insee):
        """Retourne le temps de trajet en secondes."""
        if origin_insee == dest_insee: return 300 # 5 min intra-communal par défaut
        return self.travel_time_matrix.get((origin_insee, dest_insee), None)

    def sample_demand(self, n_agents):
        """Echantillonne n_agents basés sur la matrice OD INSEE."""
        probs = self.od_matrix_df['count'] / self.od_matrix_df['count'].sum()
        sampled_indices = np.random.choice(self.od_matrix_df.index, size=n_agents, p=probs)
        sampled_trips = self.od_matrix_df.loc[sampled_indices]
        return sampled_trips[['insee_origin', 'insee_destination']].values.tolist()

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_all()
    # Test de proximité : Communes à 2 km de Paris 1er (75101)
    nearby = loader.get_nearby_communes("75101", 2.0)
    print(f"\nCommunes à moins de 2km de 75101 : {nearby}")
