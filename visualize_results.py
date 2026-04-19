import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from data_loader import DataLoader # Pour récupérer les coordonnées

def generate_visualizations(input_path="idf_simulation/results/agent_results.parquet", output_dir="plots/idf_analysis"):
    """Generates detailed visualization graphs including distance and intra-commune analysis."""
    
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} does not exist yet.")
        return

    print(f"Loading data and calculating distances...")
    df = pd.read_parquet(input_path)
    
    # Load coordinates to calculate distances
    dl = DataLoader()
    dl.load_all()
    
    # Vectorized Distance Calculation (Faster for 1M+ agents)
    coords_origin = np.array([dl.insee_to_coords.get(o, (0,0)) for o in df['origin']])
    coords_dest = np.array([dl.insee_to_coords.get(d, (0,0)) for d in df['destination']])
    
    df['distance_km'] = np.sqrt(np.sum((coords_origin - coords_dest)**2, axis=1)) / 1000.0
    df['is_intra'] = df['origin'] == df['destination']
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- 1. CARPOOLING PROBABILITY VS DISTANCE ---
    plt.figure(figsize=(10, 6))
    df['dist_bin'] = pd.cut(df['distance_km'], bins=np.arange(0, 50, 2))
    prob_dist = df.groupby('dist_bin', observed=True)['mode'].apply(lambda x: (x == 'carpool').mean() * 100).reset_index()
    prob_dist['dist_center'] = prob_dist['dist_bin'].apply(lambda x: x.mid)
    
    sns.lineplot(data=prob_dist, x='dist_center', y='mode', marker='o', color='blue', linewidth=2.5)
    plt.axhline(y=50, color='red', linestyle='--', label='50\% Threshold')
    plt.title("Carpooling Adoption Probability vs. Trip Distance (Small Crown)", fontsize=14)
    plt.xlabel("Distance (km)")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/05_carpool_prob_vs_distance.png", dpi=150) # Réduction DPI pour rapidité
    plt.close()

    # --- 2. INTRA VS INTER COMMUNE ANALYSIS ---
    plt.figure(figsize=(10, 6))
    df['Trip Type'] = df['is_intra'].map({True: 'Intra-commune', False: 'Inter-commune'})
    success_intra = df.groupby('Trip Type', observed=True)['mode'].apply(lambda x: (x == 'carpool').mean() * 100).reset_index()
    
    sns.barplot(data=success_intra, x='Trip Type', y='mode', hue='Trip Type', palette="viridis", legend=False)
    plt.title("Carpooling Success Rate: Intra vs. Inter-commune", fontsize=14)
    plt.ylabel("Success Rate (%)")
    plt.savefig(f"{output_dir}/06_intra_vs_inter_success.png", dpi=150)
    plt.close()

    # --- 3. TRAVEL TIME DISTRIBUTION (STAYING AS IS BUT RE-GENERATING) ---
    plt.figure(figsize=(10, 6))
    # On échantillonne si N > 100k pour garder le KDE lisible
    df_plot = df.sample(min(len(df), 200000)) if len(df) > 200000 else df
    sns.histplot(df_plot['baseline_travel_time']/60, label='Baseline (All Walk)', color='red', alpha=0.3, kde=True)
    sns.histplot(df_plot['travel_time']/60, label='With Carpooling', color='green', alpha=0.5, kde=True)
    plt.title(f"Impact on Population Travel Time Distribution (Sample size: {len(df_plot)})", fontsize=14)
    plt.xlabel("Travel Time (min)")
    plt.legend()
    plt.savefig(f"{output_dir}/01_travel_time_distribution.png", dpi=150)
    plt.close()

    # --- 4. SDC AND TOTAL COST (RE-GENERATING) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='mode', y='sdc', hue='mode', palette="Set2", legend=False)
    plt.title("Schedule Delay Cost (SDC): Carpool Reliability", fontsize=14)
    plt.savefig(f"{output_dir}/03_sdc_dispersion.png", dpi=150)
    plt.close()

    print(f"Analysis completed. New distance-based graphs available in: {output_dir}")

if __name__ == "__main__":
    # Ensure idf_simulation is in path for data_loader import
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'idf_simulation'))
    generate_visualizations()
