import pandas as pd
import os

# Configuration des fichiers
DATA_DIR = "data"
FILES = [
    "communes_medoids.parquet",
    "communes_free_flow_travel_times.parquet",
    "od_matrix_insee.parquet",
    "od_matrix_iris.parquet"
]

def inspect_parquet_files():
    print(f"{'='*60}")
    print(f"  INSPECTION DES DATASETS - ÎLE-DE-FRANCE")
    print(f"{'='*60}\n")

    for file_name in FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f" [!] Fichier manquant : {file_path}")
            continue

        try:
            # Chargement du fichier
            df = pd.read_parquet(file_path)
            
            print(f"--- FICHIER : {file_name} ---")
            print(f" -> Taille      : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
            print(f" -> Colonnes    : {df.columns.tolist()}")
            
            print("\n [ Schéma / Types ]")
            print(df.dtypes)
            
            print("\n [ Aperçu (2 premières lignes) ]")
            print(df.head(2))

            # Statistiques spécifiques selon le type de fichier
            if 'count' in df.columns:
                total_trips = df['count'].sum()
                print(f"\n [ Stats ] Nombre total de trajets (somme de 'count') : {total_trips:,}")
            
            if 'free_flow_travel_time' in df.columns:
                min_time = df['free_flow_travel_time'].min() / 60
                max_time = df['free_flow_travel_time'].max() / 60
                avg_time = df['free_flow_travel_time'].mean() / 60
                print(f"\n [ Stats ] Temps de trajet (min/max/moy) : {min_time:.1f} / {max_time:.1f} / {avg_time:.1f} minutes")

            print(f"\n{'-'*60}\n")

        except Exception as e:
            print(f" [X] Erreur lors de la lecture de {file_name} : {e}\n")

if __name__ == "__main__":
    inspect_parquet_files()
