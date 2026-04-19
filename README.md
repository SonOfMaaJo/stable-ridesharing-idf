# Temporal Matching for Decentralized Ridesharing Systems

This project implements a dynamic temporal matching framework for decentralized ridesharing systems, specifically applied to the **Île-de-France (IDF)** road network.

## 🔬 Overview
The model reconciles decentralized individual decision-making with stable assignment theory using an adapted **Gale-Shapley deferred acceptance mechanism**. It accounts for:
- Dynamic vehicle capacities
- Schedule delay costs (SDC)
- User compatibility constraints
- Real-world spatial networks (OSMNX)

## 📁 Data Requirements
**Note:** The raw simulation data files (`.parquet` format) are confidential and are not included in this repository. 
To run the simulations, you must provide your own data in the `data/` and `idf_simulation/data/` directories, including:
- `od_matrix_insee.parquet`
- `od_matrix_iris.parquet`
- `communes_medoids.parquet`
- `communes_free_flow_travel_times.parquet`

## 🚀 Key Features
- **Scalability:** Handles large-scale networks with nearly linear computational growth.
- **Duality Analysis:** Incorporates shadow price analysis for seat scarcity and passenger rent.
- **Stability:** Ensures the absence of blocking pairs in the matching equilibrium.

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Network Analysis:** NetworkX, OSMNX
- **Data Handling:** Pandas, NumPy, Parquet
- **Visualization:** Matplotlib, Seaborn

*Developed by Viery Naoussi under the direction of Pr. André de Palma (Thema, CY Cergy Paris Université).*
