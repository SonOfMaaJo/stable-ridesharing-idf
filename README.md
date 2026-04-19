# Temporal Matching for Decentralized Ridesharing Systems

This project implements a dynamic temporal matching framework for decentralized ridesharing systems, specifically applied to the **Île-de-France (IDF)** road network.

## 🔬 Overview
The model reconciles decentralized individual decision-making with stable assignment theory using an adapted **Gale-Shapley deferred acceptance mechanism**. It accounts for:
- Dynamic vehicle capacities
- Schedule delay costs (SDC)
- User compatibility constraints
- Real-world spatial networks (OSMNX)

## 🚀 Key Features
- **Scalability:** Handles large-scale networks with nearly linear computational growth.
- **Duality Analysis:** Incorporates shadow price analysis for seat scarcity and passenger rent.
- **Stability:** Ensures the absence of blocking pairs in the matching equilibrium.

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Network Analysis:** NetworkX, OSMNX
- **Data Handling:** Pandas, NumPy, Parquet
- **Visualization:** Matplotlib, Seaborn

## 📊 Results
Preliminary simulations on the IDF network demonstrate the efficiency of the stable matching approach in improving vehicle occupancy rates while respecting individual temporal constraints.

*Developed by Viery Naoussi under the direction of Pr. André de Palma (Thema, CY Cergy Paris Université).*
