# analytique_sante.py - VERSION CORRIGÉE

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

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

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
        r"TDR positifs": "TDR Paludisme Positifs",
        r"TOTAUX MORBIDITE": "Morbidité Totale",
        r"DECES": "Décès",
        r"Cas référés": "Référés",
        r"Femmes.*vaccinées.*VAT": "Femmes Vaccinées VAT"
}
SERVICES_CLES = list(set(SERVICE_STANDARD.values()))

# Fonction pour le calcul de la MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calcule l'erreur absolue moyenne en pourcentage (MAPE).
    Ignorer les valeurs vraies (y_true) égales à zéro pour éviter la division par zéro.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.nan # Retourne NaN si toutes les valeurs sont zéro
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

class AnalytiqueSante:
    
    SERVICE_STANDARD: Dict[str, str] = SERVICE_STANDARD
    SERVICES_CLES: List[str] = SERVICES_CLES
    
    def __init__(self):
        self.logger = logging.getLogger('analytique_sante')
        self.df_data: Optional[pd.DataFrame] = None
        self.annee_donnees: Optional[str] = None
        self.service_counts: Dict[str, int] = {}
        self.resultats_prevision: Dict[str, Any] = {}
        self.donnees_mensuelles: Optional[pd.DataFrame] = None
        self.logger.info("Instance AnalytiqueSante initialisée avec succès")

    def _standardiser_nom_service(self, service: str) -> str:
        service_str = str(service).strip().upper()
        for pattern, standard_name in self.SERVICE_STANDARD.items():
            if re.match(pattern.upper(), service_str):
                return standard_name
        return service

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
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Correction ici pour filtrer les données en fonction de l'année cible
            if annee_cible:
                self.annee_donnees = annee_cible
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

    def _preparer_donnees_ts(self, service: str) -> Optional[pd.DataFrame]:
        df_service = self.df_data[self.df_data['SERVICE'] == service].copy()
        df_service = df_service.groupby('ds')['y'].sum().reset_index()

        if df_service.empty or len(df_service) < 5:
            self.logger.warning(f"Service '{service}': {len(df_service)} points - Trop peu de données")
            return None
        
        variance = df_service['y'].var()
        if variance < 0.1:
            self.logger.warning(f"Service '{service}': Variance trop faible ({variance:.3f})")
            return None
        
        df_service = df_service.sort_values(by='ds')
        
        if len(df_service) > 1:
            full_date_range = pd.date_range(
                start=df_service['ds'].min(), 
                end=df_service['ds'].max(), 
                freq='MS'
            )
            df_full = pd.DataFrame({'ds': full_date_range})
            df_service = pd.merge(df_full, df_service, on='ds', how='left')
            df_service['y'] = df_service['y'].fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Service '{service}' préparé: {len(df_service)} points, variance={variance:.2f}")
        return df_service
        
    def _evaluer_qualite_modele(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """Évalue la qualité du modèle avec plusieurs métriques."""
        if len(y_true) < 3:
            return {'r2': -1.0, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'error': "Données insuffisantes"}
        
        y_pred_clean = np.maximum(y_pred, 0)
        
        try:
            if y_true.var() < 1e-8:
                return {'r2': -1.0, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'error': "Variance nulle"}
            
            mae = mean_absolute_error(y_true, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_clean))
            r2 = r2_score(y_true, y_pred_clean)
            mape = mean_absolute_percentage_error(y_true, y_pred_clean)
            
            if r2 > 0.98 and len(y_true) < 15:
                self.logger.warning(f"🚨 OVERFITTING DÉTECTÉ pour R²={r2:.3f} avec {len(y_true)} points")
                penalty = 0.3 + (0.2 * (15 - len(y_true)) / 15)
                r2_adjusted = max(0.1, r2 - penalty)
                
                return {
                    'r2': r2_adjusted, 
                    'mae': mae, 
                    'rmse': rmse,
                    'mape': mape,
                    'overfitting_detected': True,
                    'r2_original': r2,
                    'penalty_applied': penalty
                }
            
            return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}
            
        except Exception as e:
            return {'error': str(e)}

    def _run_prophet(self, df_ts: pd.DataFrame, service: str) -> Optional[Dict[str, Any]]:
        try:
            m = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=0.1,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.8
            )
            
            m.fit(df_ts)
            future = m.make_future_dataframe(periods=12, freq='MS')
            forecast = m.predict(future)
            
            forecast['yhat'] = np.maximum(forecast['yhat'], 0)
            
            train_pred = forecast['yhat'].iloc[:len(df_ts)]
            metrics = self._evaluer_qualite_modele(df_ts['y'], train_pred)
            
            future_forecast = forecast.iloc[len(df_ts):][['ds', 'yhat']].reset_index(drop=True)
            
            return {'model': m, 'forecast': future_forecast, 'metrics': metrics}
        
        except Exception as e:
            self.logger.error(f"Erreur Prophet pour '{service}': {e}")
            return None
            
    def _run_arima(self, df_ts: pd.DataFrame, service: str) -> Optional[Dict[str, Any]]:
        try:
            df_arima = df_ts.set_index('ds').copy()
            
            model = ARIMA(df_arima['y'], order=(1,1,1))
            model_fit = model.fit()
            
            forecast_steps = 12
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_df = forecast.summary_frame()
            forecast_df['ds'] = forecast_df.index
            forecast_df.rename(columns={'mean': 'yhat'}, inplace=True)
            forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)

            combined_df = pd.concat([df_arima.reset_index(), forecast_df], axis=0).reset_index(drop=True)
            combined_df.rename(columns={'index':'ds'}, inplace=True)
            combined_df['yhat'] = combined_df['yhat'].fillna(combined_df['y'])
            
            metrics = self._evaluer_qualite_modele(df_ts['y'], combined_df['yhat'].iloc[:len(df_ts)])
            
            return {'model': model_fit, 'forecast': forecast_df, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Erreur dans le modèle ARIMA: {e}")
            return None

    def _run_random_forest(self, df_ts: pd.DataFrame, service: str) -> Optional[Dict[str, Any]]:
        try:
            df_rf = df_ts.copy()
            df_rf['month'] = df_rf['ds'].dt.month
            df_rf['year'] = df_rf['ds'].dt.year
            
            X = df_rf[['month', 'year']]
            y = df_rf['y']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            last_date = df_ts['ds'].max()
            future_dates = []
            current_year = last_date.year
            current_month = last_date.month
            
            for i in range(12):
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
                future_dates.append(pd.Timestamp(year=current_year, month=current_month, day=1))
            
            future_df = pd.DataFrame({'ds': future_dates})
            future_df['month'] = future_df['ds'].dt.month
            future_df['year'] = future_df['ds'].dt.year
            
            predictions = model.predict(future_df[['month', 'year']])
            predictions_cleaned = np.maximum(predictions, 0)
            
            forecast = pd.DataFrame({
                'ds': future_dates, 
                'yhat': predictions_cleaned
            })
            
            metrics = self._evaluer_qualite_modele(y, model.predict(X))
            
            return {'model': model, 'forecast': forecast, 'metrics': metrics}
            
        except Exception as e:
            self.logger.error(f"Erreur RandomForest pour '{service}': {e}")
            return None

    def calculer_previsions(self) -> Dict[str, Any]:
        self.logger.info("Calcul des prévisions multi-modèles...")
        self.resultats_prevision = {}
        services_a_analyser = self.df_data['SERVICE'].unique()
        self.logger.info(f"Début de l'analyse pour {len(services_a_analyser)} services")
        
        for service in services_a_analyser:
            df_ts = self._preparer_donnees_ts(service)
            if df_ts is None:
                self.resultats_prevision[service] = {'error': "Données insuffisantes ou pas de variation pour la modélisation."}
                self.logger.warning(f"Aucun modèle n'a pu être évalué pour '{service}'.")
                continue

            models_results = {}
            best_model_name = None
            best_r2 = -float('inf')

            self.logger.info(f"Prévision pour le service: {service}")
            
            prophet_res = self._run_prophet(df_ts, service)
            if prophet_res:
                models_results['Prophet'] = prophet_res
                if 'r2' in prophet_res['metrics'] and prophet_res['metrics']['r2'] > best_r2:
                    best_r2 = prophet_res['metrics']['r2']
                    best_model_name = 'Prophet'
            
            arima_res = self._run_arima(df_ts, service)
            if arima_res:
                models_results['ARIMA'] = arima_res
                if 'r2' in arima_res['metrics'] and arima_res['metrics']['r2'] > best_r2:
                    best_r2 = arima_res['metrics']['r2']
                    best_model_name = 'ARIMA'
                    
            rf_res = self._run_random_forest(df_ts, service)
            if rf_res:
                models_results['RandomForest'] = rf_res
                if 'r2' in rf_res['metrics'] and rf_res['metrics']['r2'] > best_r2:
                    best_r2 = rf_res['metrics']['r2']
                    best_model_name = 'RandomForest'

            if best_model_name:
                self.logger.info(f"🏆 Meilleur modèle pour '{service}': {best_model_name} (R² = {best_r2:.2f})")
                self.resultats_prevision[service] = {
                    'donnees_historiques': df_ts.copy(),
                    'models': models_results,
                    'meilleur_modele': best_model_name
                }
            else:
                self.logger.warning(f"Aucun modèle viable n'a été trouvé pour le service: {service}")
                self.resultats_prevision[service] = {'error': "Aucun modèle viable trouvé."}
        
        self.logger.info("Calcul des prévisions terminé.")
        return self.resultats_prevision

    def calculer_kpis(self) -> Dict[str, Any]:
        kpis = {}
        self.logger.info("Calcul des KPIs et recommandations d'optimisation")
        
        for service, resultats in self.resultats_prevision.items():
            if 'error' in resultats:
                kpis[service] = {
                    'Volume historique annuel': 0,
                    'Volume prévu': 0,
                    'Croissance prévue': np.nan,
                    'Consultants supplémentaires estimés': 0.0
                }
                continue

            df_hist = resultats['donnees_historiques']
            df_forecast = resultats['models'][resultats['meilleur_modele']]['forecast']
            
            volume_historique = df_hist['y'].sum()
            volume_previsions = df_forecast['yhat'].sum()
            
            croissance_prevue = ((volume_previsions - volume_historique) / volume_historique) * 100 if volume_historique != 0 else np.inf
            
            taux_consultations = 1000
            if volume_previsions > volume_historique:
                consultants_suppl_estimes = (volume_previsions - volume_historique) / taux_consultations
            else:
                consultants_suppl_estimes = 0.0
                
            kpis[service] = {
                'Volume historique annuel': volume_historique,
                'Volume prévu': volume_previsions,
                'Croissance prévue': croissance_prevue,
                'Consultants supplémentaires estimés': consultants_suppl_estimes
            }
            
        return kpis

    def generer_rapport_excel_complet(self, output_stream: BinaryIO) -> None:
        try:
            with pd.ExcelWriter(output_stream, engine='xlsxwriter') as writer:
                # Feuille 1: Prévisions Détaillées par service
                for service, data in self.resultats_prevision.items():
                    if 'error' in data: 
                        continue
                    df_hist = data['donnees_historiques'].rename(columns={'ds': 'Date', 'y': 'Historique'}).set_index('Date')
                    
                    all_forecasts = pd.DataFrame({'Date': data['models']['Prophet']['forecast']['ds']}).set_index('Date')
                    for model_name, model_data in data['models'].items():
                        if not model_data['forecast'].empty:
                            df_f = model_data['forecast'].rename(columns={'ds': 'Date', 'yhat': f'Prevision_{model_name}'}).set_index('Date')
                            all_forecasts = all_forecasts.join(df_f[[f'Prevision_{model_name}']])
                    
                    combined_df = pd.concat([df_hist, all_forecasts], axis=0)
                    combined_df.to_excel(writer, sheet_name=service[:30])
                
                # Feuille 2: KPIs & Optimisation
                df_kpis = pd.DataFrame.from_dict(self.calculer_kpis(), orient='index')
                df_kpis.to_excel(writer, sheet_name='KPIs & Optimisation')

                # Feuille 3: Métriques d'Évaluation
                df_resume = pd.DataFrame()
                for service, data in self.resultats_prevision.items():
                    if 'error' not in data:
                        for model_name, model_data in data['models'].items():
                            metrics = model_data['metrics']
                            df_resume = pd.concat([df_resume, pd.DataFrame([{
                                "Service": service,
                                "Modèle": model_name,
                                "R2 du Modèle": metrics.get('r2'),
                                "MAE du Modèle": metrics.get('mae'),
                                "RMSE du Modèle": metrics.get('rmse'),
                                "MAPE du Modèle": metrics.get('mape'),
                            }])], ignore_index=True)
                
                if not df_resume.empty:
                    df_resume.to_excel(writer, sheet_name='Résumé des Métriques', index=False)
                
            self.logger.info("Rapport Excel généré avec succès.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport Excel: {e}")
            raise

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
                best_model_name = data['meilleur_modele']
                best_model_data = data['models'][best_model_name]
                
                previsions_formatted[service] = {
                    'donnees_historiques': data['donnees_historiques'],
                    'models': data['models'],
                    'meilleur_modele': {
                        'nom': best_model_name,
                        'forecast': best_model_data['forecast'],
                        'metrics': best_model_data['metrics'] # Inclure toutes les métriques
                    }
                }
            else:
                previsions_formatted[service] = data
        
        return previsions_formatted