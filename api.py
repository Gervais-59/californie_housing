from time import asctime

from fastapi import FastAPI,UploadFile, File,HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import logging,io

# On configure le comportement global (l'heure, le nom, le niveau, le message)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# On instancie le logger pour ce fichier spécifique
logger = logging.getLogger(__name__)

logger.info("Test de l'application réussi !")
app = FastAPI(title="Californie_Housing")

# 1. Téléchargement des modèles et scaler
def model_loader():
    try:
        model = joblib.load("best_model.joblib")
        scaler = joblib.load("scaler.joblib")
        logger.info(f"modèle et scaler chargés avec succès")
    except FileNotFoundError as e:
        model = None
        scaler = None
        logger.error(f"Attention, l'un des fichiers (modèle ou scaler) est introuvable: {e}")
        raise
    return model,scaler
model,scaler=model_loader()

# 2. Récupération des noms des features (il faut stocker le résultat dans une variable)
try:
    with open("features_names.json") as f:
        features_names = json.load(f)
except FileNotFoundError:
    features_names = None
    logger.error("Noms des variables introuvables")

# 3. Le contrat de données
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


# 4. La route de prédiction (Attention à la syntaxe des guillemets)
@app.post("/prediction")
def pred(X: HouseData):
    """Convertit la base en dictionnaire puis en DataFrame, applique le scaler, et prédit."""

    # Étape A : Transformation en DataFrame
    X_dic = X.model_dump()
    X_data = pd.DataFrame([X_dic])
    if model is None:
        logger.error("Prédiction demandée alors que le modèle n'est pas chargé")
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    try:
        # (Optionnel) S'assurer que l'ordre des colonnes est le même que lors de l'entraînement
        if features_names:
            X_data = X_data[features_names]


        # Le modèle a été entraîné sur des données scalées, il faut faire pareil ici.
        if scaler:
            X_data_scaled = scaler.transform(X_data)
        else:
            X_data_scaled = X_data  # Au cas où le scaler n'aurait pas chargé

        # Prédiction
        # on donne le DataFrame scalé au modèle, pas l'objet 'X' de FastAPI
        y_pred = model.predict(X_data_scaled)

        # Formater la réponse
        # y_pred est un tableau numpy (ex: [2.456]). Il nous faut extraire la valeur avec [0]
        # et la convertir en float classique pour que FastAPI puisse la renvoyer en JSON.
        prediction_finale = float(y_pred[0])
        logger.info("prédiction correctement effectuée")
        return {
            "donnees_recues": X_dic,
            "prix_predit": prediction_finale
        }

    except Exception as e:
        logger.error(f"impossible de préidre:{e}")
        raise


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):
    content = await file.read()
    X_data = pd.read_csv(io.BytesIO(content))
    if model is None:
        logger.error("Prédiction demandée alors que le modèle n'est pas chargé")
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    try:
        # Réordonner les colonnes comme à l'entraînement (sécurité)
        if features_names:
            X_data = X_data[features_names]

        X_data_scaled = scaler.transform(X_data)
        y_pred = model.predict(X_data_scaled)

        X_data = X_data.copy()
        X_data['prix_predit'] = y_pred
        logger.info("données correctement chargées")
        return X_data.to_dict(orient="records")
    except Exception as e:
        logger.error(f"impossible de charger les données:{e}")
        raise