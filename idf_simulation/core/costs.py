import numpy as np

# Paramètres économiques (Section 4 du PDF)
# Valeur du temps (Value of Time - VOT) converties de €/h en €/s
ALPHA_C = 6.0 / 3600   # Coût unitaire transport motorisé (6.0 €/h)
ALPHA_M = 10.0 / 3600  # Coût unitaire marche (10.0 €/h - effort plus élevé)
# Paramètres SDC (Section 3 du PDF) - Valeurs fournies en €/h converties en €/s
ALPHA_SDC = 6.0 / 3600 # Pour le modèle symétrique (6.0 €/h)
BETA = 4.0 / 3600      # Pénalité avance (4.0 €/h)
GAMMA = 15.0 / 3600    # Pénalité retard (15.0 €/h)
THETA = 2.0            # Pénalité fixe de retard (Section 3.4)
COORDINATION_COST = 2.0 # Coût fixe de transaction/coordination (€)

def calculate_sdc(t_arrival, t_star, model="asymmetric_linear"):
    """
    Calcule le Schedule Delay Cost (Section 3 du PDF).
    """
    delay = t_arrival - t_star

    if model == "asymmetric_linear":
        if delay > 0:
            return GAMMA * delay # Retard (15 €/h)
        else:
            return BETA * abs(delay) # Avance (4 €/h)

    elif model == "linear_symmetric":
        return ALPHA_SDC * abs(delay) # Symétrique (6 €/h)

    elif model == "quadratic":
        # Section 3.2
        return 0.0001 * (delay**2) # Pénalise lourdement les grands écarts

    elif model == "fixed_penalty":
        # Section 3.4
        sdc = GAMMA * delay if delay > 0 else BETA * abs(delay)
        if delay > 0: sdc += THETA
        return sdc

    return 0

def calculate_walking_cost(tt_m, t_star, current_time, sdc_model="asymmetric_linear"):
    """
    Coût du trajet marche C^M (Section 4.2).
    """
    t_arrival = current_time + tt_m
    return (ALPHA_M * tt_m) + calculate_sdc(t_arrival, t_star, sdc_model)

def calculate_ridesharing_cost(tt_c, t_star, current_time, tt_access=0, tt_egress=0, sdc_model="asymmetric_linear"):
    """
    Coût de covoiturage C^c (Section 4.1).
    Intègre désormais un coût fixe de transaction/coordination (Section 4.3).
    """
    t_arrival = current_time + tt_access + tt_c + tt_egress
    base_cost = (ALPHA_C * tt_c) + calculate_sdc(t_arrival, t_star, sdc_model) + (ALPHA_M * (tt_access + tt_egress))
    return base_cost + COORDINATION_COST
