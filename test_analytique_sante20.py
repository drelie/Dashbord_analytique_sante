# test_analytique_sante.py

"""
Tests unitaires et d'intégration pour le module analytique_sante.

Ce module utilise la bibliothèque pytest pour valider les fonctionnalités de
la classe AnalytiqueSante, incluant le chargement des données, la préparation,
le calcul des KPIs et la gestion des dates.
"""
import pytest
import pandas as pd
import numpy as np
import os
from io import StringIO, BytesIO
from typing import IO
from analytique_sante import AnalytiqueSante, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error
import pickle
from datetime import datetime

# On définit un dictionnaire de services qui inclut des noms standardisés
# et des noms qui doivent être transformés pour la standardisation
SERVICE_MAPPING = {
    "Nb Consultants": "Nb Consultants",
    "Nb Consultations": "Nb Consultations",
    "Accouchements en établissement": "Accouchements",
    "Total Paludisme": "Paludisme",
    "Service Inconnu": "Service Inconnu"
}

@pytest.fixture
def analytique_instance() -> AnalytiqueSante:
    """
    Fixture qui fournit une instance de AnalytiqueSante pour les tests.

    Returns:
        AnalytiqueSante: Une nouvelle instance de la classe d'analyse.
    """
    return AnalytiqueSante()


@pytest.fixture
def sample_data_complete() -> pd.DataFrame:
    """
    Crée un DataFrame de données complet avec 24 mois de données fictives.
    """
    dates = pd.to_datetime(pd.date_range(start='2022-01-31', periods=24, freq='ME'))
    services = ['Nb Consultations'] * 24
    valeurs = np.arange(100, 124) + np.random.rand(24) * 10
    
    return pd.DataFrame({
        'date': dates,
        'SERVICE': services,
        'valeur': valeurs
    })

def test_chargement_et_preparation_donnees(analytique_instance: AnalytiqueSante, sample_data_complete: pd.DataFrame) -> None:
    """
    Teste le chargement, la standardisation et la préparation des données.
    """
    csv_data = StringIO()
    sample_data_complete.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    success = analytique_instance.charger_donnees(csv_data)
    
    assert success
    assert analytique_instance.df_data is not None
    assert 'ds' in analytique_instance.df_data.columns
    assert 'y' in analytique_instance.df_data.columns
    assert 'SERVICE' in analytique_instance.df_data.columns
    assert len(analytique_instance.df_data) > 0
    assert analytique_instance.df_data['SERVICE'].iloc[0] == 'Nb Consultations'
    print("✅ Chargement et préparation des données fonctionne correctement")


def test_gestion_donnees_insuffisantes(analytique_instance: AnalytiqueSante) -> None:
    """
    Teste si le script gère correctement les services avec des données historiques insuffisantes (< 12 mois).
    """
    dates = pd.to_datetime(pd.date_range(start='2024-01-31', periods=10, freq='ME'))
    services = ['Consultations PF'] * 10
    valeurs = np.arange(1, 11)
    
    df_data = pd.DataFrame({
        'date': dates,
        'SERVICE': services,
        'valeur': valeurs
    })
    
    csv_data = StringIO()
    df_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    analytique_instance.charger_donnees(csv_data)
    resultats = analytique_instance.calculer_previsions()
    
    assert 'Consultations PF' in resultats
    assert 'error' in resultats['Consultations PF']
    assert "Données historiques insuffisantes (< 12 mois)" in resultats['Consultations PF']['error']
    
    print("✅ Gestion des données insuffisantes pour l'analyse fonctionne correctement")


def test_standardiser_nom_service() -> None:
    """
    Teste la standardisation des noms de services.
    
    Vérifie que les noms de services sont correctement mappés selon la
    configuration de la classe et que les noms non reconnus restent inchangés.
    """
    analytique_instance = AnalytiqueSante()
    
    assert analytique_instance._standardiser_nom_service("Nombre de consultants") == "Nb Consultants"
    assert analytique_instance._standardiser_nom_service("TOTAL PALUDISME") == "Paludisme"
    assert analytique_instance._standardiser_nom_service("Accouchement en établissement") == "Accouchements"
    
    assert analytique_instance._standardiser_nom_service("Service Inconnu") == "Service Inconnu"
    
    print("✅ Standardisation des noms de services fonctionne correctement")


def test_calculer_score_composite() -> None:
    """
    Teste la fonction de calcul du score composite pour s'assurer que le sMAPE est utilisé
    correctement et que le MAPE est ignoré.
    """
    analytique_instance = AnalytiqueSante()

    # Créer un dictionnaire de métriques pour le test
    # On met une valeur distincte pour MAPE et sMAPE
    metrics = {
        'r2': 0.8,
        'mae': 15.0,
        'rmse': 20.0,
        'mape': 100.0,  # Valeur volontairement élevée pour vérifier qu'elle n'est pas utilisée
        'smape': 5.0
    }

    # Calcul manuel du score composite avec les poids par défaut
    # Poids: r2=0.25, mae=0.25, rmse=0.25, smape=0.25
    r2_norm = 0.8
    mae_norm = 1 / (1 + 15.0)
    rmse_norm = 1 / (1 + 20.0)
    smape_norm = 1 / (1 + 5.0 / 100) # Division par 100 car le smape est en %

    expected_score = (
        0.25 * r2_norm +
        0.25 * mae_norm +
        0.25 * rmse_norm +
        0.25 * smape_norm
    )
    
    calculated_score = analytique_instance._calculer_score_composite(metrics)

    # Assurer que le score calculé est très proche du score attendu
    assert np.isclose(calculated_score, expected_score)
    
    print("✅ Calcul du score composite avec sMAPE fonctionne correctement")


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """
    Cette fixture s'exécute avant et après chaque test.
    """
    print(f"\n--- Début du test à {datetime.now().isoformat()} ---")
    yield
    print(f"--- Fin du test à {datetime.now().isoformat()} ---\n")