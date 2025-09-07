import pandas as pd
import numpy as np
import re
import warnings
import logging
from typing import Union, IO, Dict, Tuple, Optional, List, Any, BinaryIO
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from io import BytesIO
import xlsxwriter
import pickle

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

# Attributs de classe (déclarés en dehors de la classe)
SERVICE_STANDARD = {
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
    r"TDR positifs": "TDR Paludisme+",
    r"TOTAUX MORBIDITE": "Morbidité Totale",
    r"DECES": "Décès",
    r"Cas référés": "Référés",
    r"Femmes.*vaccinées.*VAT": "Femmes Vaccinées VAT"
}
SERVICES_CLES = list(SERVICE_STANDARD.values())

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcule l'erreur de pourcentage absolue moyenne (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mape

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calcule l'erreur de pourcentage absolue moyenne symétrique (sMAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Évite la division par zéro
    smape_values = np.zeros_like(y_true, dtype=float)
    non_zero_indices = denominator != 0
    smape_values[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
    
    return np.mean(smape_values) * 100

class AnalytiqueSante:
    """
    Classe pour l'analyse prédictive et la génération de rapports sur
    les données de santé.
    """
    SERVICE_STANDARD = SERVICE_STANDARD
    SERVICES_CLES = SERVICES_CLES

    def __init__(self, services_cles: Optional[List[str]] = None) -> None:
        """Initialise la classe AnalytiqueSante avec les attributs nécessaires."""
        self.logger = logging.getLogger(__name__)
        self.df_data: Optional[pd.DataFrame] = None
        self.predictions: Dict[str, Any] = {}
        self.annee_donnees: Optional[Union[int, str]] = None
        self.services_cles = services_cles if services_cles is not None else self.SERVICES_CLES
        self.service_counts: Dict[str, int] = {}
        self.donnees_mensuelles: Optional[pd.DataFrame] = None
        self.annee_cible: Optional[int] = None
        self.resultats_prevision: Dict[str, Any] = {}
        self.contexte_sanitaire: bool = True  # Valeur par défaut
        
        # Initialisation des modèles disponibles
        self.modeles_disponibles = {
            'Prophet': self._modele_prophet,
            'RandomForest': self._modele_random_forest,
            'ARIMA': self._modele_arima
        }

    def _standardiser_nom_service(self, service_nom: str) -> str:
        """Standardise un nom de service en utilisant les expressions régulières."""
        nom_nettoye = str(service_nom).strip()
        for pattern, standard_name in self.SERVICE_STANDARD.items():
            if re.search(pattern, nom_nettoye, re.IGNORECASE):
                return standard_name
        return nom_nettoye

    def _calculer_score_composite(self, metrics: Dict[str, Any]) -> float:
        """
        Calcule le score composite pondéré pour un ensemble de métriques.
        """
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', float('inf'))
        rmse = metrics.get('rmse', float('inf'))
        smape = metrics.get('smape', float('inf'))
        
        if pd.isna(r2) or pd.isna(mae) or pd.isna(rmse) or pd.isna(smape):
            return -float('inf')
        
        r2_norm = max(0, min(1, r2))
        mae_norm = 1 / (1 + mae) if mae != float('inf') else 0
        rmse_norm = 1 / (1 + rmse) if rmse != float('inf') else 0
        smape_norm = 1 / (1 + smape / 100) if smape != float('inf') else 0
        
        if self.contexte_sanitaire:
            poids = {'smape': 0.40, 'mae': 0.30, 'r2': 0.20, 'rmse': 0.10}
        else:
            poids = {'smape': 0.25, 'mae': 0.25, 'r2': 0.25, 'rmse': 0.25}
        
        score_composite = (
            poids['r2'] * r2_norm +
            poids['mae'] * mae_norm +
            poids['rmse'] * rmse_norm +
            poids['smape'] * smape_norm
        )
        
        return score_composite

    def charger_donnees(self, data_stream: Union[str, IO[Any]], annee_cible: Optional[int] = None) -> bool:
        self.logger.info(f"Début du chargement des données depuis: {data_stream}")
        try:
            df = pd.read_csv(data_stream)
            self.logger.info(f"Données CSV chargées: {len(df)} lignes, colonnes: {list(df.columns)}")
            
            required_columns = ['date', 'SERVICE', 'valeur']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Colonnes manquantes: {missing_columns}")
                return False
                
            if df.empty:
                self.logger.error("Le fichier CSV est empty.")
                return False
            
            df['date'] = pd.to_datetime(df['date'])
            
            if annee_cible:
                self.annee_cible = annee_cible
                df = df[df['date'].dt.year < annee_cible].copy()
                if df.empty:
                    self.logger.error(f"Aucune donnée trouvée avant l'année cible {annee_cible}.")
                    return False
            else:
                self.annee_donnees = str(df['date'].dt.year.max())
                
            self.logger.info(f"Année de prévision: {self.annee_donnees}")
            
            self.logger.info("Début de la préparation des données")
            df['SERVICE'] = df['SERVICE'].apply(self._standardiser_nom_service)
            
            df.rename(columns={'date': 'ds', 'valeur': 'y'}, inplace=True)
            df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
            
            self.df_data = df[df['SERVICE'].isin(self.SERVICES_CLES)].copy()
            
            if self.df_data.empty:
                self.logger.warning("Aucun service clé trouvé dans les données après le filtrage.")
                return False
            
            self.df_data = self.df_data.sort_values(by=['SERVICE', 'ds'])
            
            self.service_counts = self.df_data['SERVICE'].value_counts().to_dict()
            self.logger.info(f"Préparation terminée: {len(self.df_data)} points de données")
            self.logger.info(f"Services trouvés: {self.df_data['SERVICE'].unique().tolist()}")
            
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Fichier non trouvé: {data_stream}")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _evaluer_modele(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calcule les métriques de performance pour les prédictions.
        """
        try:
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred)
            smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
            return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape}
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors de l'évaluation du modèle: {e}")
            return {'r2': None, 'mae': None, 'rmse': None, 'mape': None, 'smape': None, 'error': str(e)}

    def _modele_prophet(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Modèle Prophet pour les prévisions."""
        try:
            m = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon, freq='ME')
            forecast = m.predict(future)
            
            y_true = df['y']
            y_pred = forecast['yhat'][:len(y_true)]
            metrics = self._evaluer_modele(y_true, y_pred)
            
            return {'forecast': forecast, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Échec du modèle Prophet: {e}")
            return {'error': str(e), 'metrics': self._evaluer_modele([], [])}


    def _modele_random_forest(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Modèle Random Forest pour les prévisions."""
        try:
            df['mois'] = df['ds'].dt.month
            df['annee'] = df['ds'].dt.year
            X = df[['mois', 'annee']]
            y = df['y']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=horizon + 1, freq='ME')[1:]
            future_df = pd.DataFrame({
                'ds': future_dates,
                'mois': future_dates.month,
                'annee': future_dates.year
            })
            
            forecast_values = model.predict(future_df[['mois', 'annee']])
            forecast = future_df.rename(columns={'ds': 'ds'})
            forecast['yhat'] = forecast_values
            
            y_pred = model.predict(X)
            metrics = self._evaluer_modele(y, y_pred)
            
            return {'forecast': forecast, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Échec du modèle Random Forest: {e}")
            return {'error': str(e), 'metrics': self._evaluer_modele([], [])}


    def _modele_arima(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Modèle ARIMA pour les prévisions."""
        try:
            model = ARIMA(df['y'], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=horizon)
            forecast_df = forecast.summary_frame().reset_index().rename(columns={'index': 'ds', 'mean': 'yhat'})
            
            y_true = df['y']
            y_pred = model_fit.predict()
            metrics = self._evaluer_modele(y_true, y_pred)
            
            return {'forecast': forecast_df, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Échec du modèle ARIMA: {e}")
            return {'error': str(e), 'metrics': self._evaluer_modele([], [])}

    def calculer_previsions(self) -> Dict[str, Any]:
        """
        Calcule les prévisions pour chaque service en utilisant tous les modèles
        et sélectionne le meilleur basé sur le score composite.
        """
        if self.df_data is None or self.df_data.empty:
            self.logger.error("❌ Données non chargées. Exécution annulée.")
            return {}
            
        services = self.df_data['SERVICE'].unique()
        resultats_globaux = {}
        
        for service in services:
            df_service = self.df_data[self.df_data['SERVICE'] == service].copy()
            df_service = df_service.sort_values('ds')
            
            if len(df_service) < 12:
                self.logger.warning(f"⚠️ Service '{service}': données historiques insuffisantes (< 12 mois), ignoré.")
                resultats_globaux[service] = {
                    'error': "Données historiques insuffisantes (< 12 mois)"
                }
                continue

            if self.annee_cible:
                derniere_date_donnees = df_service['ds'].iloc[-1]
                annee_derniere_date = derniere_date_donnees.year
                mois_derniere_date = derniere_date_donnees.month
                
                horizon_annee = self.annee_cible - annee_derniere_date
                horizon_mois = (horizon_annee * 12) - mois_derniere_date
                horizon = max(1, horizon_mois)
            else:
                horizon = 12
            
            resultats_service = {
                'donnees_historiques': df_service,
                'models': {},
                'meilleur_modele': None,
                'score_composite': -float('inf')
            }
            
            meilleur_score = -float('inf')
            meilleur_modele_actuel = None
            
            for nom_modele, func_modele in self.modeles_disponibles.items():
                try:
                    modele_resultat = func_modele(df_service, horizon)
                    score = self._calculer_score_composite(modele_resultat['metrics'])
                    
                    if score > meilleur_score:
                        meilleur_score = score
                        meilleur_modele_actuel = nom_modele
                        
                    modele_resultat['score_composite'] = score
                    resultats_service['models'][nom_modele] = modele_resultat
                    
                    self.logger.info(f"✅ Service: {service}, Modèle: {nom_modele}, Score: {score:.3f}")
                except Exception as e:
                    resultats_service['models'][nom_modele] = {
                        'error': f"Échec de l'exécution: {str(e)}",
                        'score_composite': -float('inf'),
                    }
                    self.logger.warning(f"⚠️ Échec du modèle {nom_modele} pour le service {service}: {e}")
            
            if meilleur_modele_actuel:
                resultats_service['meilleur_modele'] = {
                    'nom': meilleur_modele_actuel,
                    'forecast': resultats_service['models'][meilleur_modele_actuel]['forecast'],
                    'metrics': resultats_service['models'][meilleur_modele_actuel]['metrics']
                }
                resultats_service['score_composite'] = meilleur_score
                resultats_globaux[service] = resultats_service
                self.logger.info(f"🏆 Meilleur modèle pour {service}: {meilleur_modele_actuel} avec un score de {meilleur_score:.3f}")
            else:
                resultats_globaux[service] = {
                    'error': "Aucun modèle n'a pu être exécuté avec succès."
                }
                
        self.resultats_prevision = resultats_globaux
        return self.resultats_prevision

    def generer_rapport_excel_complet(self, output_file: BinaryIO) -> None:
        """
        Génère un rapport Excel complet avec plusieurs feuilles.
        """
        if not self.resultats_prevision:
            raise ValueError("Aucun résultat de prévision à générer.")

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df_resume = pd.DataFrame()
            for service, data in self.resultats_prevision.items():
                if 'error' not in data:
                    meilleur_modele_nom = data['meilleur_modele']['nom']
                    metrics = data['meilleur_modele']['metrics']
                    df_resume = pd.concat([df_resume, pd.DataFrame([{
                        "Service": service,
                        "Meilleur Modèle": meilleur_modele_nom,
                        "Score Composite": data.get('score_composite'),
                        "R2": metrics.get('r2'),
                        "MAE": metrics.get('mae'),
                        "RMSE": metrics.get('rmse'),
                        "sMAPE": metrics.get('smape'),
                    }])], ignore_index=True)
            
            if not df_resume.empty:
                df_resume = df_resume.sort_values('Score Composite', ascending=False)
                df_resume.to_excel(writer, sheet_name='Résumé des Meilleurs Modèles', index=False)
            
            df_evaluation = pd.DataFrame()
            for service, data in self.resultats_prevision.items():
                if 'error' not in data:
                    for model_name, model_data in data['models'].items():
                        if 'error' not in model_data:
                            metrics = model_data['metrics']
                            score = model_data['score_composite']
                            is_best = (model_name == data['meilleur_modele']['nom'])
                            df_evaluation = pd.concat([df_evaluation, pd.DataFrame([{
                                "Service": service,
                                "Modèle": model_name,
                                "Score Composite": score,
                                "Sélectionné": "Oui" if is_best else "Non",
                                "R2": metrics.get('r2'),
                                "MAE": metrics.get('mae'),
                                "RMSE": metrics.get('rmse'),
                                "sMAPE": metrics.get('smape'),
                            }])], ignore_index=True)
            
            if not df_evaluation.empty:
                df_evaluation = df_evaluation.sort_values(['Service', 'Score Composite'], ascending=[True, False])
                df_evaluation.to_excel(writer, sheet_name='Comparaison Modèles', index=False)
            
            df_toutes_previsions = pd.DataFrame()
            for service, data in self.resultats_prevision.items():
                if 'error' not in data:
                    df_temp = data['meilleur_modele']['forecast'].copy()
                    df_temp['Service'] = service
                    df_temp = df_temp.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})
                    df_toutes_previsions = pd.concat([df_toutes_previsions, df_temp[['Date', 'Service', 'Prévision']]], ignore_index=True)
            
            if not df_toutes_previsions.empty:
                df_toutes_previsions.to_excel(writer, sheet_name='Prévisions Détaillées', index=False)
            
        self.logger.info("✅ Rapport Excel généré avec succès.")
    
    def charger_et_preparer_donnees(self, fichier) -> None:
        success = self.charger_donnees(fichier)
        if success:
            self.donnees_mensuelles = self.df_data.rename(columns={
                'ds': 'date',
                'SERVICE': 'service',
                'y': 'valeur'
            })
        else:
            self.donnees_mensuelles = pd.DataFrame()

    def prevision_demande(self, periodes: int = 12) -> Dict[str, Any]:
        if self.df_data is None or self.df_data.empty:
            return {}
        
        resultats = self.calculer_previsions()
        
        previsions_formatted = {}
        for service, data in resultats.items():
            if 'error' not in data:
                best_model_name = data['meilleur_modele']['nom']
                best_model_data = data['models'][best_model_name]
                
                previsions_formatted[service] = {
                    'donnees_historiques': data['donnees_historiques'],
                    'models': data['models'],
                    'meilleur_modele': {
                        'nom': best_model_name,
                        'forecast': best_model_data['forecast'],
                        'metrics': best_model_data['metrics']
                    },
                    'score_composite': data.get('score_composite')  # Ajout du score composite
                }
            else:
                previsions_formatted[service] = data
        
        self.resultats_prevision = previsions_formatted
        return self.resultats_prevision