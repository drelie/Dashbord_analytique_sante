"""
Moteur d'Analyse d'Optimisation des Ressources de Santé
VERSION FINALE : Implémentation des modèles ARIMA, Prophet et RandomForest

Ce module fournit une classe AnalytiqueSante pour l'analyse prédictive
des données de santé, incluant la préparation des données, la modélisation
multi-modèles et l'évaluation des performances.

Classes:
    AnalytiqueSante: Classe principale pour l'analyse des données de santé

Exemples:
    >>> analytique = AnalytiqueSante()
    >>> analytique.charger_et_preparer_donnees("donnees_2023.csv")
    >>> resultats = analytique.prevision_demande(periodes=12)
"""

import pandas as pd
import numpy as np
import re
import warnings
import logging
from typing import Union, IO, Dict, Tuple, Optional, List, Any
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

class AnalytiqueSante:
    """
    Analyse centralisée des données de santé pour l'optimisation des ressources.
    
    Cette classe implémente un pipeline complet d'analyse prédictive incluant :
    - Préparation et nettoyage des données
    - Modélisation multi-modèles (Prophet, ARIMA, RandomForest)
    - Évaluation des performances
    - Génération de rapports
    
    Attributes:
        SERVICE_STANDARD (Dict[str, str]): Mapping des noms de services vers des standards
        SERVICES_CLES (List[str]): Liste des services clés à analyser
        donnees_brutes (Optional[pd.DataFrame]): Données brutes chargées
        donnees_mensuelles (Optional[pd.DataFrame]): Données transformées en format mensuel
        resultats_prevision (Dict[str, Any]): Résultats des prévisions par service
        annee_donnees (Optional[int]): Année des données analysées
        source_nom (str): Nom de la source de données
        logger (logging.Logger): Logger pour le suivi des opérations
    
    Example:
        >>> analytique = AnalytiqueSante()
        >>> analytique.charger_et_preparer_donnees("LBS_matrice_2023.csv")
        >>> previsions = analytique.prevision_demande(periodes=12)
        >>> analytique.generer_rapport_excel("rapport_2023.xlsx")
    """
    
    # Mapping de standardisation des noms de services avec regex
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

    SERVICES_CLES: List[str] = list(SERVICE_STANDARD.values())

    def __init__(self) -> None:
        """
        Initialise une nouvelle instance d'AnalytiqueSante.
        
        Initialise tous les attributs nécessaires pour l'analyse des données
        de santé et configure le logger pour le suivi des opérations.
        """
        self.donnees_brutes: Optional[pd.DataFrame] = None
        self.donnees_mensuelles: Optional[pd.DataFrame] = None
        self.resultats_prevision: Dict[str, Any] = {}
        self.annee_donnees: Optional[int] = None
        self.source_nom: str = "inconnu"
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Instance AnalytiqueSante initialisée avec succès")

    def _standardiser_nom_service(self, nom: str) -> str:
        """
        Standardise le nom d'un service selon le mapping défini.
        
        Args:
            nom (str): Nom du service à standardiser
            
        Returns:
            str: Nom standardisé du service
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique._standardiser_nom_service("Nombre de consultants")
            'Nb Consultants'
            >>> analytique._standardiser_nom_service("TOTAL PALUDISME")
            'Paludisme'
        """
        nom = str(nom).strip()
        for pattern, standard in self.SERVICE_STANDARD.items():
            if re.match(pattern, nom, re.IGNORECASE):
                return standard
        return nom

    def charger_et_preparer_donnees(self, source: Union[str, IO]) -> None:
        """
        Charge et prépare les données de santé pour l'analyse.
        
        Cette méthode effectue les opérations suivantes :
        1. Chargement des données depuis un fichier CSV
        2. Extraction de l'année depuis le nom du fichier
        3. Préparation et transformation des données
        4. Validation de la structure des données
        
        Args:
            source (Union[str, IO]): Chemin vers le fichier CSV ou objet file-like
            
        Raises:
            Exception: Si le chargement ou la préparation échoue
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique.charger_et_preparer_donnees("LBS_matrice_2023.csv")
        """
        self.logger.info(f"Début du chargement des données depuis: {source}")
        
        try:
            # Chargement des données
            if isinstance(source, str):
                self.donnees_brutes = pd.read_csv(source)
                self.source_nom = source
            else:
                self.donnees_brutes = pd.read_csv(source)
                self.source_nom = getattr(source, 'name', 'inconnu')

            # Extraction de l'année depuis le nom du fichier
            match = re.search(r'(\d{4})', self.source_nom)
            self.annee_donnees = int(match.group(1)) if match else pd.Timestamp.now().year
            
            self.logger.info(f"Année détectée: {self.annee_donnees}")

            # Création d'une copie pour éviter les modifications accidentelles
            try:
                self.donnees_brutes = self.donnees_brutes.copy()
            except Exception as e:
                raise Exception(f"Erreur lors du chargement des données : {e}")

            # Préparation des données
            self._preparer_donnees()
            
            self.logger.info(f"Données chargées avec succès: {len(self.donnees_mensuelles)} enregistrements")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}")
            raise

    def _preparer_donnees(self) -> None:
        """
        Prépare et transforme les données brutes en format mensuel.
        
        Cette méthode privée effectue :
        1. Standardisation des noms de colonnes
        2. Filtrage des services clés
        3. Transformation en format long (melt)
        4. Conversion des dates et valeurs
        5. Nettoyage des données
        
        Raises:
            Exception: Si la colonne 'SERVICE' est manquante ou si la transformation échoue
        """
        self.logger.info("Début de la préparation des données")
        
        df = self.donnees_brutes.copy()
        
        # Standardisation des noms de colonnes
        df.columns = [str(c).strip().upper() for c in df.columns]

        # Validation de la présence de la colonne SERVICE
        if 'SERVICE' not in df.columns:
            raise Exception("Colonne 'SERVICE' manquante dans le fichier.")

        # Standardisation des noms de services
        df['service'] = df['SERVICE'].apply(self._standardiser_nom_service)
        df = df[df['service'].isin(self.SERVICES_CLES)]

        # Mapping des mois français vers numériques
        mois_map = {
            'JANVIER': 1, 'FEVRIER': 2, 'MARS': 3, 'AVRIL': 4, 'MAI': 5, 'JUIN': 6,
            'JUILLET': 7, 'AOÛT': 8, 'SEPTEMBRE': 9, 'OCTOBRE': 10, 'NOVEMBRE': 11, 'DÉCEMBRE': 12
        }

        # Transformation en format long
        df_long = df.melt(id_vars=['service'], var_name='mois', value_name='valeur')
        df_long['mois_normalise'] = df_long['mois'].str.upper().str.strip()
        df_long = df_long[df_long['mois_normalise'].isin(mois_map.keys())]
        df_long['mois_num'] = df_long['mois_normalise'].map(mois_map)
        
        # Conversion des valeurs en numérique
        df_long['valeur'] = pd.to_numeric(df_long['valeur'], errors='coerce').fillna(0)
        
        # Création des dates
        df_long['date'] = pd.to_datetime(
            df_long.apply(lambda r: f"{self.annee_donnees}-{r['mois_num']}-01", axis=1),
            errors='coerce'
        )

        # Finalisation du dataset
        self.donnees_mensuelles = df_long.dropna(subset=['date'])[['service', 'date', 'valeur']].sort_values('date')
        
        self.logger.info(f"Préparation terminée: {len(self.donnees_mensuelles)} points de données")

    def _evaluer_qualite_modele(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Optional[float]]:
        """
        Évalue la qualité d'un modèle de prévision.
        
        Calcule plusieurs métriques de performance :
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Square Error)
        - R² (Coefficient de détermination)
        - MAPE (Mean Absolute Percentage Error)
        
        Args:
            y_true (pd.Series): Valeurs réelles
            y_pred (pd.Series): Valeurs prédites
            
        Returns:
            Dict[str, Optional[float]]: Dictionnaire contenant les métriques calculées
            
        Example:
            >>> y_true = pd.Series([100, 110, 120, 130])
            >>> y_pred = pd.Series([105, 108, 118, 132])
            >>> metrics = analytique._evaluer_qualite_modele(y_true, y_pred)
            >>> print(f"R²: {metrics['r2']:.3f}")
        """
        # Vérification de la validité des données
        if len(y_true) < 2 or y_true.nunique() == 1:
            return {'mae': None, 'rmse': None, 'r2': None, 'mape': None}
        
        # Protection contre la division par zéro
        y_true_safe = y_true.copy()
        y_true_safe[y_true_safe == 0] = 1e-6
        
        # Calcul des métriques
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

    def _creer_dates_futures(self, df_hist: pd.DataFrame, periodes: int) -> pd.DatetimeIndex:
        """
        Crée une série de dates futures pour les prévisions.
        
        Args:
            df_hist (pd.DataFrame): Données historiques avec colonne 'ds'
            periodes (int): Nombre de périodes à prévoir
            
        Returns:
            pd.DatetimeIndex: Série de dates futures
        """
        last_date = df_hist['ds'].max()
        return pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periodes, freq='MS')

    def _modele_prophet(self, df_hist: pd.DataFrame, periodes: int) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]:
        """
        Implémente le modèle Prophet pour la prévision de séries temporelles.
        
        Prophet est un modèle additif qui gère automatiquement :
        - Les tendances non-linéaires
        - La saisonnalité annuelle, hebdomadaire et quotidienne
        - Les effets des jours fériés
        - Les changements de tendance
        
        Args:
            df_hist (pd.DataFrame): Données historiques avec colonnes 'ds' et 'y'
            periodes (int): Nombre de périodes à prévoir
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]: 
                - DataFrame des prévisions
                - Métriques de performance
                - Modèle entraîné
                
        Example:
            >>> df_hist = pd.DataFrame({
            ...     'ds': pd.date_range('2023-01-01', periods=12, freq='MS'),
            ...     'y': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]
            ... })
            >>> forecast, metrics, model = analytique._modele_prophet(df_hist, 6)
        """
        try:
            # Préparation des données pour Prophet
            df_prophet = df_hist.rename(columns={'ds': 'ds', 'y': 'y'})
            
            # Configuration du modèle Prophet
            model = Prophet(
                yearly_seasonality=True,      # Saisonnalité annuelle
                weekly_seasonality=False,     # Pas de saisonnalité hebdomadaire
                daily_seasonality=False,      # Pas de saisonnalité quotidienne
                changepoint_prior_scale=0.05  # Flexibilité des changements de tendance
            )
            
            # Entraînement du modèle
            model.fit(df_prophet)
            
            # Création des données futures
            future = model.make_future_dataframe(periods=periodes, freq='MS')
            
            # Prédiction
            forecast = model.predict(future)
            
            # Extraction des prévisions et intervalles de confiance
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-periodes:].reset_index(drop=True)
            
            # Évaluation de la qualité
            metrics = self._evaluer_qualite_modele(
                df_prophet['y'], 
                model.predict(df_prophet)[['yhat']].values.flatten()
            )
            
            return forecast_result, metrics, model
            
        except Exception as e:
            self.logger.error(f"Erreur dans le modèle Prophet: {str(e)}")
            raise

    def _modele_arima(self, df_hist: pd.DataFrame, periodes: int) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]:
        """
        Implémente le modèle ARIMA pour la prévision de séries temporelles.
        
        ARIMA (AutoRegressive Integrated Moving Average) est un modèle statistique
        classique qui combine :
        - Auto-régression (AR)
        - Différenciation (I)
        - Moyenne mobile (MA)
        
        Args:
            df_hist (pd.DataFrame): Données historiques avec colonnes 'ds' et 'y'
            periodes (int): Nombre de périodes à prévoir
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]: 
                - DataFrame des prévisions
                - Métriques de performance
                - Modèle entraîné
        """
        try:
            # Extraction de la série temporelle
            y = df_hist['y'].values
            
            # Configuration du modèle ARIMA(1,1,1)
            model = ARIMA(y, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Prédiction
            forecast_values = model_fit.forecast(steps=periodes)
            
            # Création des dates futures
            dates_futures = self._creer_dates_futures(df_hist, periodes)
            
            # Construction du DataFrame de résultats
            df_forecast = pd.DataFrame({
                'ds': dates_futures, 
                'yhat': forecast_values
            })
            
            # Évaluation de la qualité
            metrics = self._evaluer_qualite_modele(df_hist['y'], model_fit.fittedvalues)
            
            return df_forecast, metrics, model_fit
            
        except Exception as e:
            self.logger.error(f"Erreur dans le modèle ARIMA: {str(e)}")
            raise

    def _modele_random_forest(self, df_hist: pd.DataFrame, periodes: int) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]:
        """
        Implémente le modèle Random Forest pour la prévision de séries temporelles.
        
        Random Forest utilise des caractéristiques dérivées du temps pour prédire
        les valeurs futures. Cette approche est particulièrement efficace pour
        capturer les patterns saisonniers et les interactions complexes.
        
        Args:
            df_hist (pd.DataFrame): Données historiques avec colonnes 'ds' et 'y'
            periodes (int): Nombre de périodes à prévoir
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Optional[float]], Any]: 
                - DataFrame des prévisions
                - Métriques de performance
                - Modèle entraîné
        """
        try:
            df = df_hist.copy()
            
            # Création des features temporelles
            df['month'] = df['ds'].dt.month
            
            # Préparation des données d'entraînement
            X = df[['month']]
            y = df['y']
            
            # Configuration et entraînement du modèle
            model = RandomForestRegressor(
                n_estimators=100,    # Nombre d'arbres
                random_state=42,     # Reproductibilité
                n_jobs=-1            # Utilisation de tous les CPU
            )
            model.fit(X, y)
            
            # Création des features pour les prévisions futures
            future_months = [(df['ds'].max() + pd.DateOffset(months=i)).month for i in range(1, periodes+1)]
            dates_futures = self._creer_dates_futures(df_hist, periodes)
            X_future = pd.DataFrame({'month': future_months})
            
            # Prédiction
            yhat = model.predict(X_future)
            
            # Construction du DataFrame de résultats
            df_forecast = pd.DataFrame({
                'ds': dates_futures, 
                'yhat': yhat
            })
            
            # Évaluation de la qualité
            metrics = self._evaluer_qualite_modele(y, model.predict(X))
            
            return df_forecast, metrics, model
            
        except Exception as e:
            self.logger.error(f"Erreur dans le modèle Random Forest: {str(e)}")
            raise

    def prevision_demande(self, periodes: int = 12) -> Dict[str, Any]:
        """
        Génère des prévisions de demande pour tous les services clés.
        
        Cette méthode implémente une approche multi-modèles :
        1. Prophet pour la saisonnalité et les tendances
        2. ARIMA pour la modélisation statistique classique
        3. Random Forest pour les patterns complexes
        4. Ensemble (moyenne) pour combiner les prévisions
        
        Args:
            periodes (int): Nombre de périodes à prévoir (défaut: 12 mois)
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant les prévisions par service
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique.charger_et_preparer_donnees("donnees.csv")
            >>> previsions = analytique.prevision_demande(periodes=6)
            >>> print(f"Prévisions générées pour {len(previsions)} services")
        """
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            self.logger.warning("Aucune donnée disponible pour les prévisions")
            return {}

        self.logger.info(f"Début des prévisions pour {periodes} périodes")
        self.resultats_prevision = {}

        for service in self.SERVICES_CLES:
            self.logger.info(f"Traitement du service: {service}")
            
            # Extraction des données du service
            df_service = self.donnees_mensuelles[self.donnees_mensuelles['service'] == service]
            if df_service.empty:
                self.logger.warning(f"Aucune donnée trouvée pour le service: {service}")
                continue

            # Préparation des données historiques
            df_hist = df_service[['date', 'valeur']].rename(columns={'date': 'ds', 'valeur': 'y'})
            models = {}

            # Modèle Prophet
            try:
                prophet_forecast, prophet_metrics, prophet_model = self._modele_prophet(df_hist, periodes)
                prophet_forecast = prophet_forecast.iloc[:periodes].reset_index(drop=True)
                models['Prophet'] = {'forecast': prophet_forecast, 'metrics': prophet_metrics, 'model': prophet_model}
                self.logger.info(f"Prophet - R²: {prophet_metrics.get('r2', 'N/A'):.3f}")
            except Exception as e:
                self.logger.error(f"Échec du modèle Prophet pour {service}: {str(e)}")
                models['Prophet'] = {'forecast': pd.DataFrame(), 'metrics': {}, 'model': None}

            # Modèle ARIMA
            try:
                arima_forecast, arima_metrics, arima_model = self._modele_arima(df_hist, periodes)
                arima_forecast = arima_forecast.iloc[:periodes].reset_index(drop=True)
                models['ARIMA'] = {'forecast': arima_forecast, 'metrics': arima_metrics, 'model': arima_model}
                self.logger.info(f"ARIMA - R²: {arima_metrics.get('r2', 'N/A'):.3f}")
            except Exception as e:
                self.logger.error(f"Échec du modèle ARIMA pour {service}: {str(e)}")
                models['ARIMA'] = {'forecast': pd.DataFrame(), 'metrics': {}, 'model': None}

            # Modèle Random Forest
            try:
                rf_forecast, rf_metrics, rf_model = self._modele_random_forest(df_hist, periodes)
                rf_forecast = rf_forecast.iloc[:periodes].reset_index(drop=True)
                models['RandomForest'] = {'forecast': rf_forecast, 'metrics': rf_metrics, 'model': rf_model}
                self.logger.info(f"RandomForest - R²: {rf_metrics.get('r2', 'N/A'):.3f}")
            except Exception as e:
                self.logger.error(f"Échec du modèle Random Forest pour {service}: {str(e)}")
                models['RandomForest'] = {'forecast': pd.DataFrame(), 'metrics': {}, 'model': None}

            # Modèle Ensemble (moyenne des modèles disponibles)
            forecasts = []
            for model_name in ['Prophet', 'ARIMA', 'RandomForest']:
                if not models[model_name]['forecast'].empty:
                    forecasts.append(models[model_name]['forecast']['yhat'])
            
            if forecasts:
                # Alignement des prévisions sur la même longueur
                min_len = min([len(f) for f in forecasts])
                forecasts = [f.iloc[:min_len].reset_index(drop=True) for f in forecasts]
                
                # Calcul de la moyenne des prévisions
                ensemble_yhat = pd.concat(forecasts, axis=1).mean(axis=1)
                ensemble_dates = models['Prophet']['forecast']['ds'].iloc[:min_len].reset_index(drop=True)
                
                df_ensemble = pd.DataFrame({
                    'ds': ensemble_dates,
                    'yhat': ensemble_yhat
                })
                models['Ensemble'] = {'forecast': df_ensemble, 'metrics': {}, 'model': None}
                self.logger.info(f"Ensemble créé avec {len(forecasts)} modèles")

            # Stockage des résultats
            self.resultats_prevision[service] = {
                'history': df_hist,
                'models': models
            }

        self.logger.info(f"Prévisions terminées pour {len(self.resultats_prevision)} services")
        return self.resultats_prevision

    def generer_rapport_excel(self, chemin_sortie: str = "rapport_sante.xlsx") -> None:
        """
        Génère un rapport Excel complet avec les résultats d'analyse.
        
        Le rapport contient :
        - Une feuille de synthèse avec les KPIs principaux
        - Des feuilles détaillées par service et modèle
        - Les métriques de performance
        - Les prévisions générées
        
        Args:
            chemin_sortie (str): Chemin du fichier Excel de sortie
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique.prevision_demande()
            >>> analytique.generer_rapport_excel("rapport_2023.xlsx")
        """
        self.logger.info(f"Génération du rapport Excel: {chemin_sortie}")
        
        try:
            writer = pd.ExcelWriter(chemin_sortie, engine='xlsxwriter')
            annee = self.annee_donnees if hasattr(self, 'annee_donnees') else pd.Timestamp.now().year

            # Feuille synthèse
            synthese = []
            for service, data in self.resultats_prevision.items():
                models = data['models']
                prophet_metrics = models.get('Prophet', {}).get('metrics', {})
                ensemble_forecast = models.get('Ensemble', {}).get('forecast', pd.DataFrame())
                
                synthese.append({
                    "Service": service,
                    "Année": annee,
                    "Prévision 1er mois": ensemble_forecast['yhat'].iloc[0] if not ensemble_forecast.empty else None,
                    "Prévision dernier mois": ensemble_forecast['yhat'].iloc[-1] if not ensemble_forecast.empty else None,
                    "R² Prophet": prophet_metrics.get('r2', None),
                    "MAE Prophet": prophet_metrics.get('mae', None),
                    "RMSE Prophet": prophet_metrics.get('rmse', None)
                })
            
            df_synthese = pd.DataFrame(synthese)
            df_synthese.to_excel(writer, sheet_name="Synthèse", index=False)

            # Feuilles détaillées par service
            for service, data in self.resultats_prevision.items():
                for model_name, model_data in data['models'].items():
                    df_forecast = model_data.get('forecast', pd.DataFrame())
                    if not df_forecast.empty:
                        sheet_name = f"{service}_{model_name[:10]}"
                        df_forecast.to_excel(writer, sheet_name=sheet_name, index=False)

            writer.close()
            self.logger.info(f"✅ Rapport généré avec succès: {chemin_sortie}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            raise

    def tableau_metriques(self) -> pd.DataFrame:
        """
        Retourne un tableau synthétique des scores des modèles pour chaque service.
        
        Returns:
            pd.DataFrame: DataFrame contenant les métriques de performance par service et modèle
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique.prevision_demande()
            >>> metriques = analytique.tableau_metriques()
            >>> print(metriques.head())
        """
        lignes = []
        for service, data in self.resultats_prevision.items():
            for model_name, model_data in data['models'].items():
                metrics = model_data.get('metrics', {})
                if metrics:
                    lignes.append({
                        "Service": service,
                        "Modèle": model_name,
                        "MAE": metrics.get('mae'),
                        "RMSE": metrics.get('rmse'),
                        "R²": metrics.get('r2'),
                        "MAPE": metrics.get('mape')
                    })
        return pd.DataFrame(lignes)

    def generer_alertes(self, seuil_croissance: float = 20.0, seuil_r2: float = 0.3) -> List[Dict[str, Any]]:
        """
        Génère des alertes métiers sur les services en forte croissance ou sous seuil de qualité.
        
        Args:
            seuil_croissance (float): Seuil de croissance en pourcentage pour déclencher une alerte
            seuil_r2 (float): Seuil minimum de R² pour la qualité du modèle
            
        Returns:
            List[Dict[str, Any]]: Liste des alertes générées
            
        Example:
            >>> analytique = AnalytiqueSante()
            >>> analytique.prevision_demande()
            >>> alertes = analytique.generer_alertes(seuil_croissance=15.0, seuil_r2=0.5)
            >>> for alerte in alertes:
            ...     print(f"Alerte: {alerte['Message']}")
        """
        alertes = []
        
        for service, data in self.resultats_prevision.items():
            # Alerte sur la croissance
            ensemble = data['models'].get('Ensemble', {}).get('forecast', pd.DataFrame())
            if not ensemble.empty and len(ensemble) > 1:
                croissance = ((ensemble['yhat'].iloc[-1] - ensemble['yhat'].iloc[0]) / (ensemble['yhat'].iloc[0] + 1e-6)) * 100
                if croissance > seuil_croissance:
                    alertes.append({
                        "Service": service,
                        "Type": "Croissance forte",
                        "Croissance (%)": round(croissance, 2),
                        "Message": f"Croissance prévue supérieure à {seuil_croissance}%"
                    })
            
            # Alerte sur la qualité du modèle
            r2 = data['models'].get('Prophet', {}).get('metrics', {}).get('r2', None)
            if r2 is not None and r2 < seuil_r2:
                alertes.append({
                    "Service": service,
                    "Type": "Qualité faible",
                    "R² Prophet": round(r2, 2),
                    "Message": f"R² du modèle Prophet inférieur à {seuil_r2}"
                })
        
        return alertes