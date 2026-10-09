# California Housing — Prédiction du prix de l'immobilier

![CI](https://github.com/Gervais-59/californie_housing/actions/workflows/CI.yml/badge.svg)

Ce projet prédit le prix médian des logements en Californie à partir de
caractéristiques comme le revenu médian du quartier, l'âge des maisons ou le
nombre moyen de pièces. Au-delà du modèle lui-même, l'objectif était de monter
un projet propre, du notebook jusqu'à une API testée et intégrée en continu.

## Ce que fait le projet

On entraîne un modèle de régression sur le jeu de données California Housing,
on le sauvegarde, puis on l'expose à travers une API web. On peut lui envoyer
les infos d'un logement et récupérer une estimation de prix, à l'unité ou par
lot (fichier CSV).

## Comment c'est organisé

Le projet suit un cycle simple mais complet. Un script d'entraînement
(`mlops.py`) prépare les données, entraîne le modèle et sauvegarde le modèle,
le scaler et la liste des features. L'API (`api.py`) charge ces fichiers et
sert les prédictions. Une suite de tests (`test_api.py`) vérifie que tout
fonctionne, et un pipeline d'intégration continue relance ces tests
automatiquement à chaque modification du code.

## Les endpoints de l'API

- `GET /` : vérifie que l'API est en ligne.
- `POST /prediction` : renvoie une estimation de prix pour un logement.
- `POST /batch_prediction` : prend un fichier CSV et renvoie une prédiction
  pour chaque ligne.

## Lancer le projet en local

Installer les dépendances :

    pip install -r requirements.txt

Entraîner le modèle (génère les fichiers nécessaires à l'API) :

    python mlops.py

Démarrer l'API :

    uvicorn api:app --reload

L'API est alors disponible sur http://localhost:8000, et sa documentation
interactive sur http://localhost:8000/docs.

## Les tests

Les tests vérifient que le modèle se charge bien, que l'API répond, qu'une
prédiction renvoie un prix valide, et que les entrées incorrectes sont gérées
proprement. Pour les lancer :

    pytest -v

## Intégration continue

À chaque push sur la branche principale, GitHub Actions démarre une machine
propre, installe les dépendances, réentraîne le modèle et lance les tests.
Le badge en haut de ce fichier indique si la dernière exécution est passée.
C'est la garantie que le projet fonctionne à partir de zéro, et pas seulement
sur ma machine.

## Stack technique

Python, scikit-learn, XGBoost, FastAPI, pytest, GitHub Actions.
## La suite, déployer ce modèle sur sur le cloud afin qu'il tourne en permanence 
