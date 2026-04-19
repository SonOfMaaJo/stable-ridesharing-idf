import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Ajout du chemin pour importer les modules de la simulation
sys.path.append(os.path.join(os.getcwd(), 'idf_simulation'))
from simulation import run_idf_simulation

def run_scalability_test():
    # On définit les points de test (Nombre de passagers)
    # On garde un ratio constant de 5:1 pour la cohérence
    passenger_counts = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    results = []

    print("Démarrage du test de scalabilité...")
    for n_p in passenger_counts:
        n_d = n_p // 5
        print(f"Test pour N = {n_p} passagers...")
        
        # On lance la simulation en mode silencieux
        kpis = run_idf_simulation(n_passengers=n_p, n_drivers=n_d, silent=True)
        
        results.append({
            "n_agents": n_p + n_d,
            "execution_time": kpis["execution_time"],
            "n_passengers": n_p
        })

    df = pd.DataFrame(results)
    
    # --- GÉNÉRATION DU GRAPHIQUE ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["n_agents"], df["execution_time"], marker='o', linestyle='-', color='red', linewidth=2)
    
    # Ajout d'une ligne de tendance théorique (O(n log n) ou O(n^2) pour comparaison)
    plt.title("Simulation Scalability: Execution Time vs. Agent Count", fontsize=14)
    plt.xlabel("Total Number of Agents (Passengers + Drivers)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, alpha=0.3)
    
    output_dir = "plots/idf_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/scalability_analysis.png", dpi=150)
    plt.close()
    
    print(f"Test terminé. Graphique sauvegardé dans : {output_dir}/scalability_analysis.png")
    print(df)

if __name__ == "__main__":
    run_scalability_test()
