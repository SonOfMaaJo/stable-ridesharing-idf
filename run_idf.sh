#!/bin/bash

# Détermine le dossier où se trouve le script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Vérification des arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_idf.sh <nombre_passagers> <nombre_conducteurs>"
    echo "Exemple: ./run_idf.sh 5000 1000"
    exit 1
fi

PASSENGERS=$1
DRIVERS=$2

# Ajoute le dossier du script au PYTHONPATH pour les imports (core, data_loader, etc.)
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

# Lancement de la simulation via le python3 du système
echo "Lancement de la simulation IDF avec $PASSENGERS passagers et $DRIVERS conducteurs..."
python3 "$SCRIPT_DIR/simulation.py" $PASSENGERS $DRIVERS
