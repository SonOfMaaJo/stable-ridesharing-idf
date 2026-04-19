from core.costs import calculate_ridesharing_cost, calculate_walking_cost

def is_compatible(profile_a, profile_b):
    """
    Fonction de compatibilité universelle.
    Par défaut : compatible si profil identique.
    Vous pourrez remplacer ceci par une matrice plus tard.
    """
    return profile_a == profile_b

class Passenger:
    """Passager cherchant un covoiturage (Section 2.1)."""
    __slots__ = ['id', 'origin_insee', 'dest_insee', 't_h', 't_star', 'status', 
                 'matched_with', 'profile', 'arrival_time', 'final_cost', 'final_sdc', 'travel_mode']

    def __init__(self, id, origin_insee, dest_insee, t_h, t_star, profile=0):
        self.id = id
        self.origin_insee = origin_insee
        self.dest_insee = dest_insee
        self.t_h = t_h         
        self.t_star = t_star   
        self.status = "waiting" 
        self.matched_with = None
        self.profile = profile 
        self.arrival_time = None
        self.final_cost = None
        self.final_sdc = None
        self.travel_mode = "walk" # Par défaut, sera "carpool" si matché

    def get_walking_cost(self, dl, current_time):
        tt_m = dl.get_travel_time(self.origin_insee, self.dest_insee)
        if tt_m is None: tt_m = 600
        else: tt_m = tt_m * 3.0 
        return calculate_walking_cost(tt_m, self.t_star, current_time)

class Driver:
    """Conducteur proposant des places (Section 2.1)."""
    __slots__ = ['id', 'origin_insee', 'dest_insee', 'capacity', 'occupants', 'occupant_profiles', 'dropoff_schedule', 'fixed_path', 'profile']

    def __init__(self, id, origin_insee, dest_insee, capacity=3, profile=0):
        self.id = id
        self.origin_insee = origin_insee
        self.dest_insee = dest_insee
        self.capacity = capacity
        self.profile = profile
        self.occupants = [] # Liste des IDs
        self.occupant_profiles = [] # Liste des profils pour vérif rapide
        self.dropoff_schedule = {} 
        self.fixed_path = [] 

    def set_fixed_path(self, dl):
        self.fixed_path = dl.get_driver_path(self.origin_insee, self.dest_insee)

    def get_available_seats(self):
        return self.capacity - len(self.occupants)

    def check_full_compatibility(self, p_profile):
        """Vérifie la compatibilité avec TOUT le véhicule (Section 4.3)."""
        # 1. Vérification avec le conducteur
        if not is_compatible(self.profile, p_profile):
            return False
        # 2. Vérification avec chaque passager déjà présent
        for o_profile in self.occupant_profiles:
            if not is_compatible(o_profile, p_profile):
                return False
        return True

    def add_passenger(self, p_id, p_profile, dropoff_time):
        if self.get_available_seats() > 0 and self.check_full_compatibility(p_profile):
            self.occupants.append(p_id)
            self.occupant_profiles.append(p_profile)
            self.dropoff_schedule[p_id] = dropoff_time
            return True
        return False

    def update_status(self, current_time):
        arrived_passengers = []
        for p_id, t_drop in list(self.dropoff_schedule.items()):
            if current_time >= t_drop:
                idx = self.occupants.index(p_id)
                self.occupants.pop(idx)
                self.occupant_profiles.pop(idx)
                del self.dropoff_schedule[p_id]
                arrived_passengers.append(p_id)
        return arrived_passengers
