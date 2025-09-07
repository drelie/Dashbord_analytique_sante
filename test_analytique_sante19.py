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
from io import StringIO
from typing import IO
from analytique_sante import AnalytiqueSante, mean_absolute_percentage_error


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
    Crée un DataFrame de données de test au format large compatible
    avec le moteur d'analyse.

    Returns:
        pd.DataFrame: Un DataFrame de données simulées.
    """
    services = [
        "Nb Consultants", "Nb Consultants <5 ans", "Nb Consultations",
        "Nb Consultations <5 ans", "Accouchements", "Naissances vivantes",
        "Paludisme", "Infections Respiratoires", "Diarrhées",
        "Dépistage Total", "Femmes VIH+", "Consultations PF",
        "TDR Paludisme Positifs", "Morbidité Totale", "Décès",
        "Référés", "Femmes Vaccinées VAT"
    ]

    np.random.seed(42)
    mois = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN',
            'JUILLET', 'AOUT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE', 'DECEMBRE']

    # Création du DataFrame au format LARGE attendu par le moteur
    data = {'SERVICE': services}
    for m in mois:
        data[m] = np.random.randint(50, 500, size=len(services))

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_csv_file(sample_data_complete: pd.DataFrame) -> IO[str]:
    """
    Crée un fichier CSV en mémoire (StringIO) avec un nom simulé,
    basé sur les données de la fixture sample_data_complete.

    Args:
        sample_data_complete (pd.DataFrame): Le DataFrame de données simulées.

    Returns:
        IO[str]: Un objet fichier en mémoire de type StringIO.
    """
    class NamedStringIO(StringIO):
        name = "test_data_2023.csv"

    output = NamedStringIO()
    sample_data_complete.to_csv(output, index=False)
    output.seek(0)
    return output


def test_charger_et_preparer_donnees(analytique_instance: AnalytiqueSante, mock_csv_file: IO[str]) -> None:
    """
    Vérifie le chargement et la préparation des données.

    Ce test s'assure que le pipeline de chargement des données fonctionne,
    que le DataFrame résultant n'est pas vide, et que les colonnes et
    les types de données sont corrects.

    Args:
        analytique_instance (AnalytiqueSante): Instance de la classe d'analyse.
        mock_csv_file (IO[str]): Fichier CSV simulé en mémoire.
    """
    analytique_instance.charger_et_preparer_donnees(mock_csv_file)

    assert not analytique_instance.donnees_mensuelles.empty
    assert 'service' in analytique_instance.donnees_mensuelles.columns
    assert 'date' in analytique_instance.donnees_mensuelles.columns
    assert 'valeur' in analytique_instance.donnees_mensuelles.columns

    services_attendus = set(analytique_instance.SERVICES_CLES)
    services_trouves = set(analytique_instance.donnees_mensuelles['service'].unique())
    assert services_attendus.issubset(services_trouves)

    assert analytique_instance.donnees_mensuelles['date'].dtype == 'datetime64[ns]'

    print(f"✅ Données chargées avec succès: {len(analytique_instance.donnees_mensuelles)} points")


def test_calculer_kpis(analytique_instance: AnalytiqueSante, mock_csv_file: IO[str]) -> None:
    """
    Teste le calcul des indicateurs de performance clés (KPIs).

    Ce test s'assure que la méthode `calculer_kpis` retourne un dictionnaire,
    ce qui indique un calcul réussi.

    Args:
        analytique_instance (AnalytiqueSante): Instance de la classe d'analyse.
        mock_csv_file (IO[str]): Fichier CSV simulé en mémoire.
    """
    analytique_instance.charger_et_preparer_donnees(mock_csv_file)
    kpis = analytique_instance.calculer_kpis()

    assert isinstance(kpis, dict)
    print(f"✅ KPIs calculés: {len(kpis)} métriques")


def test_evaluer_qualite_modele(analytique_instance: AnalytiqueSante) -> None:
    """
    Teste l'évaluation de la qualité des modèles.

    Vérifie que la méthode `_evaluer_qualite_modele` calcule correctement
    les métriques pour des données valides et gère les cas où les données
    sont insuffisantes.

    Args:
        analytique_instance (AnalytiqueSante): Instance de la classe d'analyse.
    """
    y_true = pd.Series([100, 110, 120, 130])
    y_pred = pd.Series([105, 108, 118, 132])

    metrics = analytique_instance._evaluer_qualite_modele(y_true, y_pred)
    
    # Vérification de la présence des 4 métriques
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics

    # Vérification que les valeurs ne sont pas nulles
    assert metrics['mae'] is not None
    assert metrics['rmse'] is not None
    assert metrics['r2'] is not None
    assert metrics['mape'] is not None

    y_true_insuffisant = pd.Series([100])
    y_pred_insuffisant = pd.Series([105])

    metrics_insuffisant = analytique_instance._evaluer_qualite_modele(y_true_insuffisant, y_pred_insuffisant)
    assert metrics_insuffisant['mae'] is None
    assert metrics_insuffisant['rmse'] is None
    assert metrics_insuffisant['r2'] is None
    assert metrics_insuffisant['mape'] is None

    print("✅ Évaluation de la qualité des modèles fonctionne correctement")


def test_standardiser_nom_service(analytique_instance: AnalytiqueSante) -> None:
    """
    Teste la standardisation des noms de services.

    Vérifie que les noms de services sont correctement mappés selon la
    configuration de la classe et que les noms non reconnus restent inchangés.

    Args:
        analytique_instance (AnalytiqueSante): Instance de la classe d'analyse.
    """
    assert analytique_instance._standardiser_nom_service("Nombre de consultants") == "Nb Consultants"
    assert analytique_instance._standardiser_nom_service("TOTAL PALUDISME") == "Paludisme"
    assert analytique_instance._standardiser_nom_service("Accouchement en établissement") == "Accouchements"

    assert analytique_instance._standardiser_nom_service("Service Inconnu") == "Service Inconnu"

    print("✅ Standardisation des noms de services fonctionne correctement")


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """
    Fixture qui assure la reproductibilité des tests en fixant la graine
    du générateur de nombres aléatoires.
    """
    np.random.seed(42)
    yield


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-s", "-vv", __file__]))