import time
import random
import sys
import os
import numpy as np
import pandas as pd
from data_loader import DataLoader
from core.agents import Passenger, Driver
from core.engine import StableMatchingEngine
from core.costs import calculate_ridesharing_cost, calculate_walking_cost, calculate_sdc
from tqdm import tqdm

def run_idf_simulation(n_passengers=1000, n_drivers=200, horizon_seconds=7200, delta_t=60, silent=False):
    """Lance la simulation avec suivi complet du cycle de vie et exportation de données agent-level."""
    dl = DataLoader()
    # On restreint à la Petite Couronne pour augmenter la densité d'agents
    dl.load_all(target_departments=["75", "92", "93", "94"])

    p_trips = dl.sample_demand(n_passengers)
    d_trips = dl.sample_demand(n_drivers)

    # Indexation des passagers
    passengers_dict = {}
    for i, (o, d) in enumerate(p_trips):
        t_h = random.randint(0, horizon_seconds // 2)
        tt = dl.get_travel_time(o, d) or 600
        t_star = t_h + tt + 900 
        profile = random.randint(0, 2)
        p_obj = Passenger(f"P{i}", o, d, t_h, t_star, profile=profile)
        passengers_dict[p_obj.id] = p_obj

    drivers_dict = {}
    total_capacity = 0
    for i, (o, d) in enumerate(d_trips):
        cap = random.randint(1, 4)
        total_capacity += cap
        d_profile = random.randint(0, 2)
        driver = Driver(f"D{i}", o, d, capacity=cap, profile=d_profile)
        driver.set_fixed_path(dl)
        drivers_dict[driver.id] = driver

    drivers = list(drivers_dict.values())
    start_wall_time = time.time()
    total_matched = 0
    arrived_count = 0
    matched_time_savings = []

    iterator = range(0, horizon_seconds, delta_t)
    if not silent: iterator = tqdm(iterator, desc="Simulation")

    for t in iterator:
        # 1. Mise à jour des conducteurs (Libération des places)
        for d in drivers:
            arrived_p_ids = d.update_status(t)
            for p_id in arrived_p_ids:
                p_obj = passengers_dict[p_id]
                p_obj.status = "arrived"
                p_obj.arrival_time = t
                arrived_count += 1

        # 2. Identification des nouveaux passagers
        p_entering = [p for p in passengers_dict.values() if t <= p.t_h < t + delta_t]
        
        if p_entering:
            engine = StableMatchingEngine(p_entering, drivers, dl, t)
            matches = engine.solve() 

            for d_id, p_info in matches.items():
                driver = drivers_dict[d_id]
                for p_id, t_drop in p_info:
                    p_obj = passengers_dict[p_id]
                    if driver.add_passenger(p_id, p_obj.profile, t_drop):
                        p_obj.status = "riding"
                        p_obj.travel_mode = "carpool"
                        p_obj.arrival_time = t_drop # Prévision, confirmée à l'arrivée réelle
                        total_matched += 1
                        
                        tt_direct_walk = (dl.get_travel_time(p_obj.origin_insee, p_obj.dest_insee) or 600) * 3.0
                        tt_rideshare = t_drop - t
                        matched_time_savings.append(tt_direct_walk - tt_rideshare)

    # --- CALCULS FINAUX ET EXPORTATION (AGENT-LEVEL) ---
    results_data = []
    for p in passengers_dict.values():
        tt_direct_m = dl.get_travel_time(p.origin_insee, p.dest_insee) or 600
        baseline_tt = tt_direct_m * 3.0  # Temps de marche total théorique (Baseline)
        
        # Calcul de la distance réelle via les coordonnées
        p1 = dl.insee_to_coords.get(p.origin_insee)
        p2 = dl.insee_to_coords.get(p.dest_insee)
        dist_km = 0.0
        if p1 and p2:
            dist_km = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / 1000.0

        if p.travel_mode == "carpool" and p.arrival_time:
            final_tt = p.arrival_time - p.t_h
            p.final_sdc = calculate_sdc(p.arrival_time, p.t_star)
            # Utilisation du temps de trajet motorisé estimé (vitesse x2 par rapport au flux libre)
            p.final_cost = calculate_ridesharing_cost(tt_direct_m/2.0, p.t_star, p.t_h)
        else:
            p.arrival_time = p.t_h + baseline_tt
            p.final_sdc = calculate_sdc(p.arrival_time, p.t_star)
            p.final_cost = calculate_walking_cost(baseline_tt, p.t_star, p.t_h)
            final_tt = baseline_tt

        results_data.append({
            "agent_id": p.id,
            "origin": p.origin_insee,
            "destination": p.dest_insee,
            "distance_km": dist_km,
            "profile": p.profile,
            "mode": p.travel_mode,
            "t_h": p.t_h,
            "t_star": p.t_star,
            "t_arrival": p.arrival_time,
            "baseline_travel_time": baseline_tt,
            "travel_time": final_tt,
            "sdc": p.final_sdc,
            "total_cost": p.final_cost
        })

    df_results = pd.DataFrame(results_data)
    os.makedirs("idf_simulation/results", exist_ok=True)
    output_path = "idf_simulation/results/agent_results.parquet"
    df_results.to_parquet(output_path)

    end_wall_time = time.time()
    kpis = {
        "n_passengers": n_passengers,
        "n_drivers": n_drivers,
        "total_capacity": total_capacity,
        "matched_count": total_matched,
        "arrived_count": arrived_count,
        "success_rate": (total_matched / n_passengers) * 100,
        "occupancy_rate": (total_matched / total_capacity) * 100 if total_capacity > 0 else 0,
        "avg_time_saved_min": np.mean(matched_time_savings)/60 if matched_time_savings else 0,
        "execution_time": end_wall_time - start_wall_time
    }
    
    if not silent:
        print(f"\n--- RÉSULTATS (P={n_passengers}, D={n_drivers}) ---")
        print(f"Exportation détaillée : {output_path}")
        print(f"Capacité totale offerte : {kpis['total_capacity']} sièges")
        print(f"Passagers matchés       : {kpis['matched_count']}")
        print(f"Taux de succès passager : {kpis['success_rate']:.2f}%")
        print(f"Gain de temps moyen     : {kpis['avg_time_saved_min']:.1f} min")
        print(f"Temps de calcul         : {kpis['execution_time']:.2f} s")
        
    return kpis

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_idf_simulation(n_passengers=p, n_drivers=d)
