import streamlit as st
import requests
import pandas as pd

BASE_URL  = "https://konan-calif-hous-ai-518773933873.europe-west9.run.app"
URL_UNIT  = f"{BASE_URL}/prediction"
URL_BATCH = f"{BASE_URL}/batch_prediction"

st.title("Prédiction du prix de l'immobilier — Californie")

onglet_unit, onglet_batch = st.tabs(["Prédiction unique", "Prédiction par lot (CSV)"])

# ONGLET 1 : prédiction unique
with onglet_unit:
    st.write("Renseigne les caractéristiques d'un secteur.")

    MedInc     = st.number_input("Revenu médian (en 10k$)", value=3.5, step=0.1)
    HouseAge   = st.number_input("Âge médian des logements", value=30.0, step=1.0)
    AveRooms   = st.number_input("Nombre moyen de pièces", value=5.0, step=0.1)
    AveBedrms  = st.number_input("Nombre moyen de chambres", value=1.0, step=0.1)
    Population = st.number_input("Population du secteur", value=1000.0, step=10.0)
    AveOccup   = st.number_input("Occupation moyenne par logement", value=3.0, step=0.1)
    Latitude   = st.number_input("Latitude", value=34.0, step=0.01)
    Longitude  = st.number_input("Longitude", value=-118.0, step=0.01)

    if st.button("Prédire le prix"):
        payload = {
            "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms,
            "AveBedrms": AveBedrms, "Population": Population, "AveOccup": AveOccup,
            "Latitude": Latitude, "Longitude": Longitude,
        }
        try:
            r = requests.post(URL_UNIT, json=payload, timeout=30)
            r.raise_for_status()
            prix = r.json()["prix_predit"]
            st.success(f"Prix médian estimé : {prix:.2f} (en centaines de milliers de $)")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

# ONGLET 2 : prédiction par lot
with onglet_batch:
    st.write("Charge un CSV avec les 8 colonnes de features. "
             "L'API prédit chaque ligne et te renvoie le fichier complété.")

    fichier = st.file_uploader("Fichier CSV", type="csv")

    if fichier is not None:
        # Aperçu de ce que l'utilisateur a chargé
        apercu = pd.read_csv(fichier)
        st.write("Aperçu du fichier :")
        st.dataframe(apercu.head())

        if st.button("Lancer la prédiction par lot"):
            try:
                # On renvoie le fichier au tout début (file_uploader a déjà lu le curseur)
                fichier.seek(0)
                files = {"file": (fichier.name, fichier.getvalue(), "text/csv")}
                r = requests.post(URL_BATCH, files=files, timeout=60)
                r.raise_for_status()

                resultats = pd.DataFrame(r.json())
                st.success(f"{len(resultats)} lignes prédites.")
                st.dataframe(resultats)

                # Bouton pour télécharger les résultats
                st.download_button(
                    "Télécharger les résultats",
                    resultats.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")