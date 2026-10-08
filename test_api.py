# test_api.py
from fastapi.testclient import TestClient
from api import app, model_loader

client = TestClient(app)


# --- Test 1 : la fonction de chargement ---
def test_model_loader_retourne_deux_objets():
    model, scaler = model_loader()
    # Si les fichiers existent, ni l'un ni l'autre ne doit être None
    assert model is not None
    assert scaler is not None


# --- Test 2 : l'endpoint d'accueil (smoke test) ---
def test_accueil_repond():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["Status"] == "ok"


# --- Test 3 : une prédiction valide ---
def test_prediction_valide():
    maison = {
        "MedInc": 8.3,
        "HouseAge": 41.0,
        "AveRooms": 6.9,
        "AveBedrms": 1.0,
        "Population": 322.0,
        "AveOccup": 2.5,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/prediction", json=maison)
    assert response.status_code == 200

    data = response.json()
    # La réponse doit contenir la clé "prix_predit"
    assert "prix_predit" in data
    # Le prix doit être un nombre positif (un prix négatif serait absurde)
    assert data["prix_predit"] > 0


# --- Test 4 : FastAPI valide les entrées ---
def test_prediction_champ_manquant():
    maison_incomplete = {"MedInc": 8.3}  # il manque 7 champs
    response = client.post("/prediction", json=maison_incomplete)
    # FastAPI doit renvoyer 422 (erreur de validation), pas 500
    assert response.status_code == 422