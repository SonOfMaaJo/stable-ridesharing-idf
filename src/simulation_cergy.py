import time
import random
from src.network_manager import NetworkManager
from src.agents import Passenger, Driver
from src.matching import StableMatching
from tqdm import tqdm

def run_dynamic_simulation(num_passengers=500, num_drivers=100, delta_t=300):
    nm = NetworkManager("Cergy, France")
    G = nm.load_or_download_graph()
    
    # 1. Génération des agents
    all_nodes = nm.get_random_nodes(num_passengers*2 + num_drivers*2, unique=False)
    
    passengers = []
    for i in range(num_passengers):
        t_h = random.randint(0, 3600) # Départ entre 0 et 1h
        t_star = t_h + 1800 + random.randint(0, 1800) # Arrivée souhaitée ~45min après
        profile = random.randint(0, 1) # 2 profils sociaux
        passengers.append(Passenger(f"P{i}", all_nodes[i*2], all_nodes[i*2+1], t_h, t_star, profile))
        
    drivers = []
    for i in range(num_drivers):
        d = Driver(f"D{i}", all_nodes[num_passengers*2+i*2], all_nodes[num_passengers*2+i*2+1], profile=random.randint(0, 1))
        d.set_fixed_path(G)
        drivers.append(d)
    
    # 2. Boucle temporelle T
    start_sim_time = time.time()
    total_matches = 0
    active_drivers = {d.id: d for d in drivers}
    total_capacity = sum(d.capacity for d in drivers)
    
    print(f"Simulation dynamique sur {num_passengers} passagers...")
    for t in tqdm(range(0, 7200, delta_t), desc="Boucle temporelle"): # Simulation sur 2 heures
        # Identifier passagers arrivant à t
        p_active_at_t = [p for p in passengers if t <= p.t_h < t + delta_t]
        
        if not p_active_at_t:
            continue
            
        sm = StableMatching(p_active_at_t, active_drivers, G, t)
        p_prefs, d_prefs = sm.build_preference_lists()
        matches = sm.solve(p_prefs, d_prefs)
        
        # Appliquer les résultats
        step_matches = 0
        for d_id, p_ids in matches.items():
            active_drivers[d_id].occupants.extend(p_ids)
            step_matches += len(p_ids)
        
        total_matches += step_matches
        # print(f"Time {t}s : {len(p_active_at_t)} passagers actifs, {step_matches} nouveaux matches.")

    end_sim_time = time.time()
    
    # 3. Affichage des Statistiques Finales
    print("\n" + "="*40)
    print(" STATISTIQUES FINALES DE SIMULATION")
    print("="*40)
    print(f"Durée totale du calcul : {end_sim_time - start_sim_time:.2f} secondes")
    print(f"Nombre total de passagers : {num_passengers}")
    print(f"Nombre total de conducteurs : {num_drivers}")
    print(f"Capacité totale théorique : {total_capacity} places")
    print("-" * 40)
    print(f"Nombre total de matches : {total_matches}")
    print(f"Nombre de passagers marchant : {num_passengers - total_matches}")
    print("-" * 40)
    matching_rate = (total_matches / num_passengers) * 100
    occupancy_rate = (total_matches / total_capacity) * 100
    print(f"Taux de Matching Passagers : {matching_rate:.2f} %")
    print(f"Taux d'Occupation Véhicules : {occupancy_rate:.2f} %")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_dynamic_simulation(num_passengers=10000, num_drivers=2000)
