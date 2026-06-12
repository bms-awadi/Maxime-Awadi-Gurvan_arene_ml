import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from preprocess import preprocess

def get_modeles():
    return {
        "Régression linéaire": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Arbre de décision": DecisionTreeRegressor(max_depth=10, random_state=0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=200, max_depth=10, random_state=42)
    }

def entrainer_et_evaluer(modele, X_train, X_test, y_train, y_test):
    """Entraîne le modèle, prédit et renvoie le R² score (coefficient de détermination)."""
    modele.fit(X_train, y_train)
    return r2_score(y_test, modele.predict(X_test))

def arene(modeles, X_train, X_test, y_train, y_test, titre="Classement"):
    """Fait s'affronter tous les modèles sur le MÊME split. Trie par R² score."""
    resultats = []
    for nom, modele in modeles.items():
        # Entraîner et évaluer le modèle directement
        r2 = entrainer_et_evaluer(modele, X_train, X_test, y_train, y_test)
        resultats.append((r2, nom))

    # Trie automatiquement par le premier élément du tuple (le score R²) du plus grand au plus petit
    resultats.sort(reverse=True)

    print(f"{titre} :")
    for rang, (r2, nom) in enumerate(resultats, start=1):
        print(f"{rang}. {nom:<24}: R² = {r2:.1%}")
    return resultats


def sauvegarder_champion(champion_modele, scaler, chemin="models/champion.joblib"):
    """Sauvegarde le modèle champion et son scaler sous forme de dictionnaire."""
    dossier = os.path.dirname(chemin)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    joblib.dump({"modele": champion_modele, "scaler": scaler}, chemin)
    print(f"Modèle champion et scaler sauvegardés dans {chemin}")

def sauvegarder_tous_les_modeles(modeles, scaler, dossier="models/baselines"):
    """Entraîne tous les modèles sur les données normalisées et les sauvegarde dans un dossier."""
    os.makedirs(dossier, exist_ok=True)
    joblib.dump(scaler, os.path.join(dossier, "scaler.joblib"))
    
    for nom, modele in modeles.items():
        nom_fichier = nom.lower().replace(" ", "_") + ".joblib"
        chemin = os.path.join(dossier, nom_fichier)
        joblib.dump(modele, chemin)
        print(f"Modèle '{nom}' sauvegardé dans {chemin}")

def main():
    print("Phase 1 : Split Train/Test")
    X_train, X_test, y_train, y_test = preprocess()
    print(f"Taille train : {X_train.shape[0]} joueurs, test : {X_test.shape[0]} joueurs")

    print("Phase 2 : l'Arène")
    modeles = get_modeles()
    classement = arene(modeles, X_train, X_test, y_train, y_test,
                       titre="Classement des modèles")
    champion_nom = classement[0][1]
    
    print("Phase 3 : Normalisation et entraînement final du champion")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    final_champion = modeles[champion_nom]
    final_champion.fit(X_train_scaled, y_train)
        
    print("Phase 4 : Sauvegarde")
    sauvegarder_champion(final_champion, scaler)

if __name__ == "__main__":
    main()