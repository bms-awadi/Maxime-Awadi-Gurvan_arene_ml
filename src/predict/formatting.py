def format_valeur(euros: float) -> str:
    """Formate une valeur en euros de façon lisible."""
    if euros >= 1_000_000:
        return f"{euros / 1_000_000:.1f}M €"
    elif euros >= 1_000:
        return f"{euros / 1_000:.0f}K €"
    return f"{euros:.0f} €"
