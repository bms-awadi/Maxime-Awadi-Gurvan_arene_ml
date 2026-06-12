from __future__ import annotations

import pandas as pd

# Import `load_data` de façon robuste : supporte l'exécution comme script
# (`python src\preprocess.py`) et l'import depuis la racine du projet
# (`from src.preprocess import preprocess`).
try:  # si nous sommes importés en tant que package (src.preprocess)
    from . import load_data
except Exception:
    try:  # exécution comme script : module sibling dans sys.path
        import load_data
    except Exception:
        # dernier recours : importer via le nom de package `src.load_data`
        try:
            import src.load_data as load_data
        except Exception:
            raise ModuleNotFoundError(
                "Could not import 'load_data'. Ensure you run from project root or install package correctly."
            )


def convert_columns_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Phase: Réparation numérique

    Pour chaque colonne fournie, on tente une conversion en numérique.
    - `errors='coerce'` transforme les valeurs non numériques en `NaN` afin
      qu'elles soient ensuite traitées explicitement (p.ex. imputation).

    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fill_missing_with_median(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Phase: Imputation

    Pour les colonnes numériques, on remplace les valeurs manquantes par la
    médiane. Ce choix est robuste face aux valeurs extrêmes et permet de
    conserver l'ensemble des observations (comme dans le notebook).
    """
    df = df.copy()
    for column in numeric_columns:
        if column in df.columns:
            median = df[column].median()
            df[column] = df[column].fillna(median)
    return df


def encode_categorical(df: pd.DataFrame, categorical_columns: list[str], drop_first: bool = True) -> pd.DataFrame:
    """
    Phase: Encodage des variables catégorielles

    - Colonnes binaires : mapping manuel possible (0/1) si on souhaite
      contrôler l'ordre. Ici nous utilisons `get_dummies` pour être
      cohérents avec l'approche One-Hot du notebook.
    - `drop_first=True` limite la redondance et l'effet miroir linéaire.
    """
    df = df.copy()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=drop_first, dtype=int)
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de préparation (regroupe plusieurs phases du notebook) :

    0) Suppression des identifiants techniques (`player_id`) qui ne sont pas
       utiles pour la modélisation.
    1) Conversion des colonnes numériques (détecte et force le type numérique).
    2) Imputation des valeurs manquantes numériques par médiane.
    3) Encodage des colonnes textuelles/catégorielles via One-Hot.

    Le code laisse volontairement certaines colonnes textuelles (p.ex. `name`)
    non encodées car elles sont plutôt de l'information descriptive.
    """
    df = df.copy()

    # Supprimer les identifiants techniques
    if "player_id" in df.columns:
        df = df.drop(columns=["player_id"])

    # Colonnes de type texte présentes dans le DataFrame
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    # conserver certaines colonnes textuelles si elles sont utiles en l'état
    for keep in ("name", "current_club_name"):
        if keep in text_columns:
            text_columns.remove(keep)

    # Colonnes numériques détectées automatiquement
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 1) Conversion explicite (permet d'exposer des valeurs déguisées en texte)
    df = convert_columns_numeric(df, numeric_columns)

    # 2) Imputation numérique
    df = fill_missing_with_median(df, numeric_columns)

    # 3) Encodage : ne pas encoder des colonnes textuelles à très haute
    #    cardinalité (URLs, codes, noms uniques, etc.) car cela explose la
    #    mémoire lors du One-Hot. On sélectionne uniquement les colonnes
    #    catégorielles ayant un nombre raisonnable de modalités.
    candidate_cats = [col for col in text_columns if col not in ("name", "current_club_name")]

    # seuil empirique : ne pas encoder si plus de 50 modalités uniques
    categorical_columns = [col for col in candidate_cats if df[col].nunique() <= 50]

    # si aucune colonne ne satisfait, on passe sans erreur
    df = encode_categorical(df, categorical_columns)

    # NOTE: on ne retire pas ici `valuation_eur` (la cible) – laisser le caller
    # décider s'il veut extraire X/y via `build_feature_target`.
    return df


def build_feature_target(df: pd.DataFrame, target_column: str = "valuation_eur") -> tuple[pd.DataFrame, pd.Series]:
    """Sépare le tableau en features `X` et cible `y`.

    Rationnel : vérifier explicitement la présence de la colonne cible pour
    éviter des erreurs silencieuses plus tard dans le pipeline.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess(data_dir: str = "data", test_size: float = 0.2, random_state: int = 42, target_column: str = "valuation_eur") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Grosse fonction de préparation qui renvoie les jeux de train/test.

    Retour dans l'ordre demandé par l'utilisateur : `Xtest, Xtrain, Ytest, Ytrain`.
    - charge les données brutes via `load_data.load_raw_dataset`
    - applique `prepare_dataset`
    - extrait `X` et `y` via `build_feature_target`
    - effectue un `train_test_split`
    """
    raw = load_data.load_raw_dataset(data_dir)
    clean = prepare_dataset(raw)
    X, y = build_feature_target(clean, target_column=target_column)

    try:
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise ImportError("scikit-learn is required for train/test split. Install it in your environment.") from exc

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Retourner dans l'ordre demandé
    return X_test, X_train, y_test, y_train


if __name__ == "__main__":
    # Exécution rapide pour vérification manuelle du pipeline
    # utiliser la nouvelle fonction `preprocess` pour vérifier l'ensemble du flux
    X_test, X_train, y_test, y_train = preprocess()
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
