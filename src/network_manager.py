import osmnx as ox
import networkx as nx
import pickle
import os

class NetworkManager:
    def __init__(self, city_name="Cergy, France"):
        self.city_name = city_name
        self.graph_path = f"data/{city_name.replace(', ', '_')}.graphml"
        self.graph = None

    def load_or_download_graph(self):
        """Charge le graphe depuis le disque ou le télécharge si nécessaire."""
        if os.path.exists(self.graph_path):
            print(f"Chargement du graphe de {self.city_name} depuis {self.graph_path}...")
            self.graph = ox.load_graphml(self.graph_path)
        else:
            print(f"Téléchargement du graphe de {self.city_name} via OSMNX...")
            if not os.path.exists("data"):
                os.makedirs("data")
            self.graph = ox.graph_from_place(self.city_name, network_type="drive")
            # On ne garde que la composante fortement connexe
            self.graph = ox.project_graph(self.graph)
            self.graph = ox.truncate.largest_component(self.graph, strongly=True)
            ox.save_graphml(self.graph, self.graph_path)
        return self.graph

    def get_shortest_path_length(self, source_node, target_node):
        """Calcule la distance la plus courte entre deux nœuds en mètres."""
        return nx.shortest_path_length(self.graph, source_node, target_node, weight="length")

    def get_random_nodes(self, n, unique=True):
        """Retourne n nœuds aléatoires du graphe."""
        import random
        nodes = list(self.graph.nodes)
        if unique:
            if n > len(nodes):
                raise ValueError(f"Demande de {n} nœuds uniques mais le graphe n'en contient que {len(nodes)}.")
            return random.sample(nodes, n)
        else:
            return random.choices(nodes, k=n)

if __name__ == "__main__":
    # Test pour Cergy
    nm = NetworkManager("Cergy, France")
    G = nm.load_or_download_graph()
    print(f"Cergy chargé : {len(G.nodes)} nœuds.")
    
    # Test pour le Val-d'Oise (Optionnel, à lancer manuellement si besoin)
    # nm_vdo = NetworkManager("Val-d'Oise, France")
    # G_vdo = nm_vdo.load_or_download_graph()
    # print(f"Val-d'Oise chargé : {len(G_vdo.nodes)} nœuds.")
