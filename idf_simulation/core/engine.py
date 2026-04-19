import multiprocessing as mp
import numpy as np
from functools import partial
from core.costs import calculate_ridesharing_cost, calculate_walking_cost
from core.agents import is_compatible

def compute_passenger_prefs_chunk(passenger_chunk, drivers_data, driver_by_commune, dl_matrix, search_radius_km, current_time, dl_obj):
    """Calcule les préférences avec un double filtrage spatial pour la performance."""
    chunk_p_prefs = {}
    chunk_p_match_data = {}
    chunk_p_driver_proximity = {} 
    radius_m = search_radius_km * 1000.0
    MIN_CARPOOL_DIST_KM = 2.0 # Seuil minimal pour envisager le covoiturage (Nouveauté)

    for p in passenger_chunk:
        options = []
        
        # Calcul de la distance directe du trajet passager
        p1 = dl_obj.insee_to_coords.get(p.origin_insee)
        p2 = dl_obj.insee_to_coords.get(p.dest_insee)
        p_dist_km = 0.0
        if p1 and p2:
            p_dist_km = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / 1000.0
        
        # Filtre de distance minimale (Section 5.1)
        if p_dist_km < MIN_CARPOOL_DIST_KM:
            chunk_p_prefs[p.id] = [None]
            chunk_p_match_data[p.id] = {}
            chunk_p_driver_proximity[p.id] = {}
            continue

        tt_direct = dl_matrix.get((p.origin_insee, p.dest_insee), 600)
        cost_walking_total = calculate_walking_cost(tt_direct * 3.0, p.t_star, current_time)
        
        # 1. Communes proches de la destination
        nearby_dest_communes = dl_obj.get_nearby_communes(p.dest_insee, search_radius_km)
        nearby_dest_set = set(nearby_dest_communes)

        # 2. Filtrage des conducteurs passant par l'origine
        origin_drivers = driver_by_commune.get(p.origin_insee, {})
        
        # 3. Intersection : conducteurs passant par l'origine ET une commune proche de la destination
        candidate_ids = set()
        for dest_c in nearby_dest_communes:
            dest_drivers = driver_by_commune.get(dest_c, {})
            # Intersection rapide via les clés du dictionnaire (driver_id)
            common = origin_drivers.keys() & dest_drivers.keys()
            for d_id in common:
                # Vérifier que la dépose est APRES la prise en charge
                if dest_drivers[d_id] >= origin_drivers[d_id]:
                    candidate_ids.add(d_id)

        for d_id in candidate_ids:
            d_origin, d_dest, d_path, d_seats, d_profile, d_occupant_profiles = drivers_data[d_id]
            
            if d_seats <= 0: continue
            if not is_compatible(d_profile, p.profile): continue
            
            compatible_with_all = True
            for o_profile in d_occupant_profiles:
                if not is_compatible(o_profile, p.profile):
                    compatible_with_all = False
                    break
            if not compatible_with_all: continue

            # Trouver le meilleur point de dépose parmi les communes du trajet qui sont "proches"
            m_o_idx = origin_drivers[d_id]
            path_after_mo = d_path[m_o_idx:]
            
            best_m_d = None
            min_egress_tt = float('inf')
            
            for c in path_after_mo:
                if c in nearby_dest_set:
                    tt_to_dest = dl_matrix.get((c, p.dest_insee), 0)
                    if tt_to_dest < min_egress_tt:
                        min_egress_tt = tt_to_dest
                        best_m_d = c
            
            if best_m_d:
                tt_mo_to_md = dl_matrix.get((p.origin_insee, best_m_d), 300)
                cost_ride = calculate_ridesharing_cost(
                    tt_mo_to_md, p.t_star, current_time, 0, min_egress_tt * 3.0
                )

                if cost_ride < cost_walking_total:
                    dropoff_time = current_time + tt_mo_to_md
                    options.append((d_id, cost_ride, dropoff_time, m_o_idx))

        options.sort(key=lambda x: x[1])
        chunk_p_prefs[p.id] = [opt[0] for opt in options] + [None]
        chunk_p_match_data[p.id] = {opt[0]: opt[2] for opt in options}
        chunk_p_driver_proximity[p.id] = {opt[0]: opt[3] for opt in options}
        
    return chunk_p_prefs, chunk_p_match_data, chunk_p_driver_proximity

class StableMatchingEngine:
    def __init__(self, passengers, drivers, data_loader, current_time, search_radius_km=3.0):
        self.passengers = {p.id: p for p in passengers}
        self.drivers = {d.id: d for d in drivers}
        self.dl = data_loader
        self.current_time = current_time
        self.search_radius_km = search_radius_km

    def build_preference_lists(self):
        # driver_by_commune[commune] = { d_id: index_in_path }
        driver_by_commune = {}
        drivers_data = {}
        for d in self.drivers.values():
            drivers_data[d.id] = (
                d.origin_insee, d.dest_insee, d.fixed_path, d.get_available_seats(),
                d.profile, list(d.occupant_profiles)
            )
            for idx, commune in enumerate(d.fixed_path):
                if commune not in driver_by_commune: driver_by_commune[commune] = {}
                driver_by_commune[commune][d.id] = idx

        p_list = list(self.passengers.values())
        num_cores = mp.cpu_count()
        chunk_size = max(1, len(p_list) // num_cores)
        chunks = [p_list[i:i + chunk_size] for i in range(0, len(p_list), chunk_size)]

        with mp.Pool(processes=num_cores) as pool:
            func = partial(compute_passenger_prefs_chunk, 
                           drivers_data=drivers_data,
                           driver_by_commune=driver_by_commune,
                           dl_matrix=self.dl.travel_time_matrix,
                           search_radius_km=self.search_radius_km,
                           current_time=self.current_time,
                           dl_obj=self.dl)
            results = pool.map(func, chunks)

        passenger_prefs = {}
        all_p_match_data = {}
        driver_prefs_scores = {d_id: {} for d_id in self.drivers}

        for p_prefs, p_match_data, p_prox in results:
            passenger_prefs.update(p_prefs)
            all_p_match_data.update(p_match_data)
            for p_id, prox_dict in p_prox.items():
                for d_id, score in prox_dict.items():
                    driver_prefs_scores[d_id][p_id] = score

        for p_id, data in all_p_match_data.items():
            self.passengers[p_id].matched_with = data

        return passenger_prefs, driver_prefs_scores

    def solve(self):
        p_prefs, d_prefs_scores = self.build_preference_lists()
        unmatched_p = list(self.passengers.keys())
        p_proposals_idx = {p_id: 0 for p_id in self.passengers}
        matches = {d_id: [] for d_id in self.drivers}
        
        while unmatched_p:
            p_id = unmatched_p.pop(0)
            prefs = p_prefs.get(p_id, [None])
            if p_proposals_idx[p_id] >= len(prefs): continue
            
            target_d_id = prefs[p_proposals_idx[p_id]]
            p_proposals_idx[p_id] += 1
            if target_d_id is None: continue
            
            current_matches = matches[target_d_id]
            current_matches.append(p_id)
            capacity = self.drivers[target_d_id].get_available_seats()
            
            if len(current_matches) > capacity:
                current_matches.sort(key=lambda x: d_prefs_scores[target_d_id][x])
                rejected_p = current_matches.pop() 
                unmatched_p.append(rejected_p)
                
        final_assignments = {}
        for d_id, p_ids in matches.items():
            final_assignments[d_id] = [(p_id, self.passengers[p_id].matched_with[d_id]) for p_id in p_ids]
        return final_assignments
