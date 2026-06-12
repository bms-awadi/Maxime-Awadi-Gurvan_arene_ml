import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page Streamlit (titre, disposition)
st.set_page_config(
    page_title="Football Player Market Value Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Insertion du dossier 'src' dans le chemin système pour pouvoir importer nos scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from predict.loader import load_champion
    from predict.predictor import predict_player, predict_by_name
    from predict.leaderboard import leaderboard_predictions
    from predict.cli import SUB_POSITIONS, CHAMPIONSHIPS
    from load_data import load_raw_dataset
    from preprocess import prepare_dataset
except ImportError as e:
    st.error(f"Erreur d'importation des modules du projet : {e}")
    st.info("Avez-vous entraîné le modèle en lançant `python src/train.py` ?")
    st.stop()


# ---------------------------------------------------------------------------
# Caching pour des performances fluides
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_cached():
    """Charge le modèle champion."""
    try:
        return load_champion()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

def safe_format_age(age_val):
    """Formate l'âge proprement en gérant les valeurs nulles."""
    if pd.isna(age_val) or age_val is None or str(age_val).strip() in ("?", "", "nan", "None"):
        return "Inconnu"
    try:
        return f"{int(float(age_val))} ans"
    except (ValueError, TypeError):
        return "Inconnu"

@st.cache_data
def load_data_cached():
    """Charge et prépare le dataset complet."""
    try:
        df_raw = load_raw_dataset()
        X = prepare_dataset(df_raw)
        X = X.copy()
        # Rattacher les colonnes nécessaires à l'affichage
        if "position" in df_raw.columns:
            X["position"] = df_raw["position"].values
        if "current_club_name" in df_raw.columns:
            X["club_name"] = df_raw["current_club_name"].values
        
        # Calcul de l'âge du joueur
        if "date_of_birth" in df_raw.columns:
            dob = pd.to_datetime(df_raw["date_of_birth"], errors="coerce")
            import datetime
            current_year = datetime.datetime.now().year
            X["age"] = current_year - dob.dt.year
        else:
            X["age"] = np.nan
            
        return X
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None


# Chargement initial des ressources
model = load_model_cached()
X_data = load_data_cached()


# ---------------------------------------------------------------------------
# Design System (CSS personnalisés pour un effet premium et moderne)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f8fafc;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #38bdf8;
        margin: 0.8rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Cartes de résultats */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .result-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .result-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #38bdf8;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barre Latérale de Navigation & Statistiques rapides
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>Navigation</h2>", unsafe_allow_html=True)
    menu = st.radio(
        "Choisissez l'onglet :",
        ["Accueil", "Recherche par Joueur", "Leaderboard Opportunités"]
    )
    
    st.markdown("---")
    st.markdown("<h3 style='margin-bottom: 0.5rem;'>Statistiques du Dataset</h3>", unsafe_allow_html=True)
    if X_data is not None:
        st.metric(label="Joueurs indexés", value=f"{len(X_data):,}")
        # Gérer le cas où l'âge peut être nul
        if "age" in X_data.columns:
            age_mean = X_data["age"].dropna().mean()
            st.metric(label="Moyenne d'âge", value=f"{age_mean:.1f} ans" if pd.notna(age_mean) else "Inconnue")
    else:
        st.warning("Données indisponibles.")


# En-tête principal commun
st.markdown("""
<div class="main-header">
    <h1>Football Player Market Value Predictor</h1>
    <p>Estimation de la valeur marchande des joueurs basée sur les statistiques réelles de Transfermarkt</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Onglet 1 : Accueil et Vue d'ensemble
# ---------------------------------------------------------------------------
if menu == "Accueil":
    st.subheader("Bienvenue sur la plateforme d'estimation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Description du Projet
        Cette application utilise un modèle de Machine Learning entraîné sur les données publiques de **Transfermarkt** pour estimer la valeur marchande d'un joueur en euros (€).
        
        Le modèle se base sur :
        - L'**âge** du joueur et son **poste**.
        - Ses **statistiques de performance** (buts, passes décisives, minutes jouées).
        - La **réputation de son championnat** actuel (basée sur le coefficient UEFA/ligue).
        """)
        
    with col2:
        logo_path = Path(__file__).resolve().parent / "Logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)


# ---------------------------------------------------------------------------
# Onglet 2 : Recherche par Joueur
# ---------------------------------------------------------------------------
elif menu == "Recherche par Joueur":
    st.subheader("Recherche d'un joueur dans la base de données")
    
    if X_data is None:
        st.error("Impossible d'effectuer des recherches car les données ne sont pas chargées.")
    else:
        # Barre de recherche avec autocomplétion basique
        search_query = st.text_input("Saisissez le nom d'un joueur (ex: Kylian Mbappé, Erling Haaland) :", "").strip()
        
        if search_query:
            # Recherche insensible à la casse
            mask = X_data["name"].str.lower().str.contains(search_query.lower(), na=False)
            matches = X_data[mask]
            
            if matches.empty:
                st.warning(f"Aucun joueur trouvé pour '{search_query}'. Essayez un autre nom.")
            else:
                # Si plusieurs joueurs correspondent, demander de choisir
                if len(matches) > 1:
                    st.info(f"Plusieurs joueurs ({len(matches)}) correspondent à votre recherche. Veuillez en sélectionner un :")
                    options = [f"{row['name']} - {row.get('club_name', 'Sans club')} ({row.get('position', 'Inconnu')})" for _, row in matches.iterrows()]
                    selected_option = st.selectbox("Sélectionnez le joueur :", options)
                    selected_idx = options.index(selected_option)
                    player_row = matches.iloc[selected_idx]
                else:
                    player_row = matches.iloc[0]
                
                # Exécuter la prédiction
                with st.spinner("Estimation de la valeur marchande..."):
                    try:
                        result = predict_by_name(model, player_row["name"], X_data)
                        
                        # Affichage des résultats
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-title">Joueur</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #f8fafc;">{result['joueur']}</div>
                                <hr style="border-color: #334155; margin: 1rem 0;">
                                <div class="result-title">Club</div>
                                <div style="font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0.8rem;">{result['club']}</div>
                                <div class="result-title">Poste</div>
                                <div style="font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0.8rem;">{result['position']}</div>
                                <div class="result-title">Âge</div>
                                <div style="font-size: 1.1rem; color: #e2e8f0;">{safe_format_age(player_row.get('age'))}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div class="result-card" style="height: 100%;">
                                <div class="result-title">Valeur Estimée par l'IA</div>
                                <div class="result-value">{result['valeur_estimee_str']}</div>
                            """, unsafe_allow_html=True)
                            
                            if "valeur_reelle_str" in result:
                                diff = result["ecart_pct"]
                                color = "#10b981" if result["statut"] == "sous-évalué" else "#ef4444" if result["statut"] == "sur-évalué" else "#f59e0b"
                                
                                st.markdown(f"""
                                <hr style="border-color: #334155; margin: 1rem 0;">
                                <div class="result-title">Valeur Marchande Actuelle (Transfermarkt)</div>
                                <div style="font-size: 1.8rem; font-weight: 600; color: #f8fafc;">{result['valeur_reelle_str']}</div>
                                <div style="margin-top: 0.8rem;">
                                    <span style="background-color: {color}; color: #ffffff; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                                        {result['statut'].upper()} ({diff:+.1f}%)
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la prédiction : {e}")


# ---------------------------------------------------------------------------
# Onglet 3 : Leaderboard Opportunités
# ---------------------------------------------------------------------------
elif menu == "Leaderboard Opportunités":
    st.subheader("Top opportunités de marché")
    st.write("Ces joueurs présentent la plus forte surévaluation théorique par rapport à leur prix de marché (potentiels joueurs sous-évalués).")
    
    if X_data is None:
        st.error("Données indisponibles.")
    else:
        # Choix de la taille du leaderboard
        top_n = st.slider("Nombre de joueurs à afficher :", min_value=5, max_value=50, value=20)
        
        with st.spinner("Calcul des prédictions sur tout le dataset..."):
            try:
                # Appel du leaderboard (save=False pour éviter d'écrire en tâche de fond Streamlit)
                df_leaderboard = leaderboard_predictions(model, X_data, top_n=top_n, save=False)
                
                # Formatage de la table
                rename_map = {
                    "name": "Joueur",
                    "club_name": "Club",
                    "position": "Poste",
                    "age": "Âge",
                    "valeur_reelle_str": "Valeur Marchande",
                    "valeur_estimee_str": "Valeur Estimée (IA)",
                    "ecart_pct": "Différence (%)"
                }
                
                cols_to_keep = [c for c in ["name", "club_name", "position", "age", "valeur_reelle_str", "valeur_estimee_str", "ecart_pct"] if c in df_leaderboard.columns]
                display_df = df_leaderboard[cols_to_keep].rename(columns=rename_map)
                
                # Formater l'âge pour gérer les valeurs nulles
                if "Âge" in display_df.columns:
                    display_df["Âge"] = display_df["Âge"].apply(safe_format_age)
                
                # Affichage
                st.dataframe(display_df, use_container_width=True)
                
                st.info("Un écart élevé indique que les statistiques de performance du joueur sont supérieures à ce que suggère sa valeur marchande actuelle.")
            except Exception as e:
                st.error(f"Erreur lors de la génération du leaderboard : {e}")
