# config.py

"""
Configuration centrale pour le système d'analyse de santé.

Ce module centralise les paramètres partagés par plusieurs scripts,
notamment les mappings pour la standardisation des services.
"""

from typing import Dict, List

# Mapping des noms de services vers des standards pour l'analyse
# Les motifs de regex sont utilisés pour une reconnaissance flexible.
SERVICE_STANDARD: Dict[str, str] = {
    r"Nombre de consultants(?!.*<)": "Nb Consultants",
    r"Nombre de consultants.*< *5": "Nb Consultants <5 ans",
    r"Nombre de consultations(?!.*<)": "Nb Consultations",
    r"Nombre de consultations.*< *5": "Nb Consultations <5 ans",
    r"Accouchement.*établissement": "Accouchements",
    r"Naissances vivantes": "Naissances vivantes",
    r"TOTAL PALUDISME": "Paludisme",
    r"TOTAL IRA": "Infections Respiratoires",
    r"TOTAL DIARRHEES": "Diarrhées",
    r"clients dépistés TOTAL": "Dépistage Total",
    r"Femme.*dépistée VIH": "Femmes VIH+",
    r"TOTAL CONSULTATION PF": "Consultations PF",
    r"TDR positifs": "TDR Paludisme Positifs",
    r"TOTAUX MORBIDITE": "Morbidité Totale",
    r"DECES": "Décès",
    r"Cas référés": "Référés",
    r"Femmes.*vaccinées.*VAT": "Femmes Vaccinées VAT"
}

# Liste des noms de services standardisés pour filtrer les données
SERVICES_CLES: List[str] = list(SERVICE_STANDARD.values())

