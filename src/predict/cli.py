import json

SUB_POSITIONS = [
    "Goalkeeper",
    "Centre-Back",
    "Left-Back",
    "Right-Back",
    "Defensive Midfield",
    "Central Midfield",
    "Left Midfield",
    "Right Midfield",
    "Left Winger",
    "Right Winger",
    "Centre-Forward",
    "Second Striker",
]

CHAMPIONSHIPS = {
    "1":  ("ES1",   "Liga (Espagne)"),
    "2":  ("L1",    "Bundesliga (Allemagne)"),
    "3":  ("FR1",   "Ligue 1 (France)"),
    "4":  ("IT1",   "Serie A (Italie)"),
    "5":  ("GB1",   "Premier League (Angleterre)"),
    "6":  ("NL1",   "Eredivisie (Pays-Bas)"),
    "7":  ("PO1",   "Liga Portugal"),
    "8":  ("BE1",   "Pro League (Belgique)"),
    "9":  ("TR1",   "Süper Lig (Turquie)"),
    "10": ("RU1",   "Premier Liga (Russie)"),
    "11": ("MLS1",  "MLS (USA)"),
    "12": ("BRA1",  "Brasileirão"),
    "13": ("ARG1",  "Liga Argentina"),
    "14": ("C1",    "Ligue des champions"),
}


def input_manual() -> dict:
    """Demande les stats d'un joueur en mode interactif."""
    print("\n" + "=" * 55)
    print("  Saisie manuelle d'un joueur")
    print("=" * 55)

    def ask_int(prompt, default=0):
        val = input(f"  {prompt} [{default}] : ").strip()
        return int(val) if val else default

    def ask_float(prompt, default=0.0):
        val = input(f"  {prompt} [{default}] : ").strip()
        return float(val) if val else default

    # Sous-position
    print("\n  Postes disponibles :")
    for i, p in enumerate(SUB_POSITIONS, 1):
        print(f"    {i:>2}. {p}")
    pos_idx = ask_int("Numéro du poste", 11) - 1
    sub_position = SUB_POSITIONS[max(0, min(pos_idx, len(SUB_POSITIONS) - 1))]

    # Pied dominant
    print("\n  Pied dominant : 1. Droit  2. Gauche")
    foot_idx = ask_int("Choix", 1)
    foot = "left" if foot_idx == 2 else "right"

    # Championnat
    print("\n  Championnats disponibles :")
    for k, (code, label) in CHAMPIONSHIPS.items():
        print(f"    {k:>2}. {label}")
    champ_idx = str(ask_int("Numéro du championnat", 5))
    championship_code = CHAMPIONSHIPS.get(champ_idx, ("GB1", ""))[0]

    stats = {
        "sub_position": sub_position,
        "foot": foot,
        "championship_code": championship_code,
        "height_in_cm": ask_int("Taille (cm)", 181),
        "international_caps": ask_int("Sélections en équipe nationale", 0),
        "international_goals": ask_int("Buts en équipe nationale", 0),
        "highest_market_value_in_eur": ask_float("Valeur marchande max historique (€)", 0),
        "last_season": ask_int("Dernière saison active (ex: 2024)", 2024),
    }

    print(f"\n  Récapitulatif : {json.dumps(stats, ensure_ascii=False, indent=4)}")
    return stats


def print_prediction(result: dict) -> None:
    print("\n" + "=" * 50)
    print("  ESTIMATION DE VALEUR MARCHANDE")
    print("=" * 50)
    if "joueur" in result:
        print(f"  Joueur   : {result['joueur']}")
    if "club" in result:
        print(f"  Club     : {result['club']}")
    if "position" in result:
        print(f"  Poste    : {result['position']}")
    if "age" in result:
        print(f"  Age      : {result['age']} ans")
    print()
    print(f"  Valeur estimée  : {result['valeur_estimee_str']}")
    if "valeur_reelle_str" in result:
        print(f"  Valeur réelle   : {result['valeur_reelle_str']}")
        print(f"  Ecart           : {result['ecart_pct']:+.1f}%  ->  {result['statut']}")
    print("=" * 50 + "\n")
