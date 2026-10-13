from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import io
import logging

# Configuration du logging, une seule fois au démarrage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Californie_Housing")

# 1. Chargement des modèles et scaler
def model_loader():
    try:
        model = joblib.load("best_model.joblib")
        scaler = joblib.load("scaler.joblib")
        logger.info("Modèle et scaler chargés avec succès")
    except FileNotFoundError:
        model = None
        scaler = None
        logger.error("Modèle ou scaler introuvable : l'API ne pourra pas prédire")
    return model, scaler

model, scaler = model_loader()

try:
    with open("features_names.json") as f:
        features_names = json.load(f)
except FileNotFoundError:
    features_names = None
    logger.warning("features_names.json introuvable : l'ordre des colonnes ne sera pas vérifié")


class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def accueil():
    return {"Status": "ok", "Message": "Bienvenue sur l'API California Housing"}


@app.post("/prediction")
def pred(X: HouseData):
    logger.info("Requête de prédiction reçue")

    # Garde-fou : si le modèle n'a pas chargé, inutile d'aller plus loin
    if model is None:
        logger.error("Prédiction demandée alors que le modèle n'est pas chargé")
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        X_dic = X.model_dump()
        X_data = pd.DataFrame([X_dic])

        if features_names:
            X_data = X_data[features_names]

        if scaler:
            X_data_scaled = scaler.transform(X_data)
        else:
            X_data_scaled = X_data

        y_pred = model.predict(X_data_scaled)
        prediction_finale = float(y_pred[0])

        # Valeur hors de la plage plausible : on continue, mais on signale
        if prediction_finale < 0 or prediction_finale > 5:
            logger.warning(f"Prédiction inhabituelle : {prediction_finale:.2f} (hors plage attendue 0-5)")

        logger.info(f"Prédiction faite : {prediction_finale:.2f}")

        return {
            "donnees_recues": X_dic,
            "prix_predit": prediction_finale,
        }

    except Exception as e:
        logger.error(f"Échec de la prédiction : {e}")
        raise