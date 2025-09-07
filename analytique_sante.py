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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
import joblib
import os

# Import des constantes depuis le fichier de configuration
from config import SERVICE_STANDARD, SERVICES_CLES

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

# Les attributs de classe SERVICE_STANDARD et SERVICES_CLES sont maintenant importés
# et ne sont plus définis ici directement.

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
    les données de santé - Version améliorée avec visualisations et caching.
    """
    # Les attributs de classe sont maintenant directement les constantes importées
    SERVICE_STANDARD = SERVICE_STANDARD
    SERVICES_CLES = SERVICES_CLES

    def __init__(self, services_cles: Optional[List[str]] = None, cache_dir: str = "cache_dir") -> None:
        """Initialise la classe AnalytiqueSante avec les attributs nécessaires."""
        self.logger = logging.getLogger(__name__)
        self.df_data: Optional[pd.DataFrame] = None
        self.predictions: Dict[str, Any] = {}
        self.annee_donnees: Optional[Union[int, str]] = None
        # Utilise self.SERVICES_CLES qui est maintenant la constante importée
        self.services_cles = services_cles if services_cles is not None else self.SERVICES_CLES
        self.service_counts: Dict[str, int] = {}
        self.donnees_mensuelles: Optional[pd.DataFrame] = None
        self.annee_cible: Optional[int] = None
        self.resultats_prevision: Dict[str, Any] = {}
        self.contexte_sanitaire: bool = True  # Valeur par défaut
        
        # Configuration du cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_enabled = True
        self.cache_ttl_hours = 24  # Cache valide pendant 24h
        
        # Initialisation des modèles disponibles
        self.modeles_disponibles = {
            'Prophet': self._modele_prophet,
            'RandomForest': self._modele_random_forest,
            'ARIMA': self._modele_arima
        }

    def _generate_cache_key(self, service: str, model_type: str, data_hash: str) -> str:
        """
        Génère une clé de cache unique basée sur le service, le type de modèle et les données.
        
        Args:
            service (str): Nom du service
            model_type (str): Type de modèle (Prophet, RandomForest, ARIMA)
            data_hash (str): Hash des données d'entraînement
            
        Returns:
            str: Clé de cache unique
        """
        cache_key = f"{service}_{model_type}_{data_hash}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """
        Génère un hash des données pour identifier les changements.
        
        Args:
            df (pd.DataFrame): Données d'entraînement
            
        Returns:
            str: Hash MD5 des données
        """
        # Utiliser les colonnes essentielles pour le hash
        data_str = f"{df['ds'].iloc[0]}_{df['ds'].iloc[-1]}_{len(df)}_{df['y'].sum():.2f}"
        return hashlib.md5(data_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Vérifie si le cache est encore valide selon le TTL.
        
        Args:
            cache_file (Path): Chemin vers le fichier de cache
            
        Returns:
            bool: True si le cache est valide, False sinon
        """
        if not cache_file.exists():
            return False
        
        # Vérifier l'âge du fichier
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() < (self.cache_ttl_hours * 3600)

    def _load_cached_model(self, service: str, model_type: str, data_hash: str) -> Optional[Dict[str, Any]]:
        """
        Charge un modèle depuis le cache s'il existe et est valide.
        
        Args:
            service (str): Nom du service
            model_type (str): Type de modèle
            data_hash (str): Hash des données
            
        Returns:
            Optional[Dict[str, Any]]: Modèle en cache ou None
        """
        if not self.cache_enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(service, model_type, data_hash)
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            
            if self._is_cache_valid(cache_file):
                self.logger.info(f"📦 Modèle {model_type} pour {service} chargé depuis le cache")
                return joblib.load(cache_file)
            else:
                self.logger.info(f"⏰ Cache expiré pour {model_type} - {service}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors du chargement du cache pour {model_type} - {service}: {e}")
            return None

    def _save_model_to_cache(self, service: str, model_type: str, data_hash: str, model_data: Dict[str, Any]) -> None:
        """
        Sauvegarde un modèle entraîné dans le cache.
        
        Args:
            service (str): Nom du service
            model_type (str): Type de modèle
            data_hash (str): Hash des données
            model_data (Dict[str, Any]): Données du modèle à sauvegarder
        """
        if not self.cache_enabled:
            return
            
        try:
            cache_key = self._generate_cache_key(service, model_type, data_hash)
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            
            # Sauvegarder le modèle
            joblib.dump(model_data, cache_file)
            self.logger.info(f"💾 Modèle {model_type} pour {service} sauvegardé dans le cache")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors de la sauvegarde du cache pour {model_type} - {service}: {e}")

    def _clear_expired_cache(self) -> None:
        """Nettoie le cache expiré pour libérer de l'espace."""
        try:
            expired_files = []
            for cache_file in self.cache_dir.glob("*.joblib"):
                if not self._is_cache_valid(cache_file):
                    expired_files.append(cache_file)
            
            for expired_file in expired_files:
                expired_file.unlink()
                
            if expired_files:
                self.logger.info(f"🧹 {len(expired_files)} fichiers de cache expirés supprimés")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors du nettoyage du cache: {e}")

    def _get_cache_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict[str, Any]: Statistiques du cache
        """
        try:
            cache_files = list(self.cache_dir.glob("*.joblib"))
            total_size = sum(f.stat().st_size for f in cache_files)
            valid_files = [f for f in cache_files if self._is_cache_valid(f)]
            
            return {
                'total_files': len(cache_files),
                'valid_files': len(valid_files),
                'expired_files': len(cache_files) - len(valid_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors du calcul des stats du cache: {e}")
            return {}

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
        Version améliorée avec bornage entre 0 et 1.
        """
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', float('inf'))
        rmse = metrics.get('rmse', float('inf'))
        smape = metrics.get('smape', float('inf'))
        
        if pd.isna(r2) or pd.isna(mae) or pd.isna(rmse) or pd.isna(smape):
            return 0.0
        
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
        
        # Amélioration 1: Forcer le bornage entre 0 et 1
        return max(0.0, min(1.0, score_composite))

    def _calculer_horizon_securise(self, df_service: pd.DataFrame) -> int:
        """
        Calcule l'horizon de prévision de manière sécurisée.
        Amélioration 2: Calcul robuste de l'horizon.
        """
        if self.annee_cible:
            derniere_date_donnees = df_service['ds'].iloc[-1]
            annee_derniere_date = derniere_date_donnees.year
            mois_derniere_date = derniere_date_donnees.month
            
            # Calcul amélioré de l'horizon
            mois_restants_annee_courante = 12 - mois_derniere_date
            annees_completes = max(0, self.annee_cible - annee_derniere_date - 1)
            horizon = mois_restants_annee_courante + (annees_completes * 12)
            
            # S'assurer que l'horizon est toujours positif
            return max(1, horizon)
        else:
            return 12

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
                self.logger.error("Le fichier CSV est vide.")
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
            forecast = future_df.copy()
            forecast['yhat'] = forecast_values
            
            y_pred = model.predict(X)
            metrics = self._evaluer_modele(y, y_pred)
            
            return {'forecast': forecast, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Échec du modèle Random Forest: {e}")
            return {'error': str(e), 'metrics': self._evaluer_modele([], [])}

    def _modele_arima(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """
        Modèle ARIMA pour les prévisions avec fallback robuste.
        Amélioration 3: ARIMA avec ordres multiples et gestion d'erreurs.
        """
        # Obtenir le nom du service au début de la fonction
        service = df['SERVICE'].iloc[0]

        ordres_arima = [(1,1,1), (0,1,1), (2,1,1), (1,1,0), (0,1,0)]
        
        for ordre in ordres_arima:
            try:
                self.logger.info(f"Tentative ARIMA avec ordre {ordre} pour le service {service}")
                model = ARIMA(df['y'], order=ordre)
                model_fit = model.fit()
                
                # Générer les prévisions
                forecast_result = model_fit.get_forecast(steps=horizon)
                forecast_df = forecast_result.summary_frame().reset_index()
                
                # Créer les dates futures
                last_date = df['ds'].iloc[-1]
                future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='ME')[1:]
                
                forecast_df['ds'] = future_dates
                forecast_df = forecast_df.rename(columns={'mean': 'yhat'})
                
                # Calculer les métriques sur les données d'entraînement
                y_true = df['y']
                y_pred = model_fit.predict(start=0, end=len(y_true)-1)
                metrics = self._evaluer_modele(y_true, y_pred)
                
                self.logger.info(f"✅ ARIMA {ordre} réussi pour le service {service}")
                return {'forecast': forecast_df[['ds', 'yhat']], 'metrics': metrics}
                
            except Exception as e:
                self.logger.warning(f"⚠️ Échec ARIMA {ordre} pour le service {service}: {e}")
                # Affiche l'erreur en clair dans le terminal pour le débogage
                print(f"⚠️ Échec ARIMA {ordre} pour le service {service}: {e}")
                continue
        
        # Si tous les ordres échouent
        self.logger.error(f"Tous les ordres ARIMA ont échoué pour le service {service}")
        return {'error': "Tous les ordres ARIMA ont échoué", 'metrics': self._evaluer_modele([], [])}

    def _generer_graphique_service(self, service: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Génère un graphique HTML pour un service donné.
        Amélioration 4: Visualisation automatique dans le rapport.
        """
        try:
            if 'error' in data:
                return None
                
            # Données historiques
            df_hist = data['donnees_historiques']
            
            # Données de prévision du meilleur modèle
            forecast_data = data['meilleur_modele']['forecast']
            meilleur_modele_nom = data['meilleur_modele']['nom']
            
            # Créer le graphique
            fig = go.Figure()
            
            # Ajouter les données historiques
            fig.add_trace(go.Scatter(
                x=df_hist['ds'], 
                y=df_hist['y'],
                mode='lines+markers',
                name='Données Historiques',
                line=dict(color='blue', width=2)
            ))
            
            # Ajouter les prévisions
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'], 
                y=forecast_data['yhat'],
                mode='lines+markers',
                name=f'Prévisions ({meilleur_modele_nom})',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Mise en forme
            fig.update_layout(
                title=f'Prévisions pour {service} - Score: {data.get("score_composite", 0):.3f}',
                xaxis_title='Date',
                yaxis_title='Valeur',
                hovermode='x unified',
                height=400
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{service.replace(' ', '_')}")
            
        except Exception as e:
            self.logger.error(f"Erreur génération graphique pour {service}: {e}")
            return None

    def calculer_previsions(self) -> Dict[str, Any]:
        """
        Calcule les prévisions pour chaque service en utilisant tous les modèles
        et sélectionne le meilleur basé sur le score composite.
        """
        if self.df_data is None or self.df_data.empty:
            self.logger.error("⛔ Données non chargées. Exécution annulée.")
            return {}
            
        services = self.df_data['SERVICE'].unique()
        resultats_globaux = {}
        
        for service in services:
            df_service = self.df_data[self.df_data['SERVICE'] == service].copy()
            df_service = df_service.sort_values('ds')
            
            if len(df_service) < 12:
                self.logger.warning(f"⚠️ Service '{service}': données historiques insuffisantes (< 12 mois), ignoré.")
                resultats_globaux[service] = {
                    'error': "Données historiques insuffisantes (< 12 mois)",
                    'score_composite': 0.0, # Assurer la présence de la clé
                    'models': {}
                }
                continue

            # Utiliser le calcul d'horizon sécurisé
            horizon = self._calculer_horizon_securise(df_service)
            
            resultats_service = {
                'donnees_historiques': df_service,
                'models': {},
                'meilleur_modele': None,
                'score_composite': 0.0  # Initialisé à 0 au lieu de -inf
            }
            
            meilleur_score = -1.0 # Utiliser une valeur négative pour garantir une première sélection
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
                    # Gérer l'échec en stockant un score de 0.0 pour garantir la cohérence des données
                    resultats_service['models'][nom_modele] = {
                        'error': f"Échec de l'exécution: {str(e)}",
                        'score_composite': 0.0,
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
                    'error': "Aucun modèle n'a pu être exécuté avec succès.",
                    'score_composite': 0.0 # Assurer la présence de la clé
                }
                
        self.resultats_prevision = resultats_globaux
        return self.resultats_prevision

    def generer_rapport_excel_complet(self, output_file: BinaryIO) -> None:
        """
        Génère un rapport Excel complet avec plusieurs feuilles et graphiques HTML.
        Version améliorée avec visualisations intégrées.
        """
        if not self.resultats_prevision:
            raise ValueError("Aucun résultat de prévision à générer.")

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            
            # Feuille 1: Résumé des modèles
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
            
            # Feuille 2: Comparaison de tous les modèles
            df_evaluation = pd.DataFrame()
            for service, data in self.resultats_prevision.items():
                if 'error' not in data:
                    for model_name, model_data in data['models'].items():
                        # Gérer les cas où le modèle a échoué
                        metrics = model_data.get('metrics', {})
                        score = model_data.get('score_composite', 0.0)
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
            
            # Feuille 3: Prévisions détaillées
            df_toutes_previsions = pd.DataFrame()
            for service, data in self.resultats_prevision.items():
                if 'error' not in data:
                    df_temp = data['meilleur_modele']['forecast'].copy()
                    df_temp['Service'] = service
                    df_temp = df_temp.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})
                    df_toutes_previsions = pd.concat([df_toutes_previsions, df_temp[['Date', 'Service', 'Prévision']]], ignore_index=True)
            
            if not df_toutes_previsions.empty:
                df_toutes_previsions.to_excel(writer, sheet_name='Prévisions Détaillées', index=False)

            # Feuille 4: Graphiques HTML (nouveau)
            graphiques_html = []
            for service, data in self.resultats_prevision.items():
                graphique_html = self._generer_graphique_service(service, data)
                if graphique_html:
                    graphiques_html.append(f"<h2>{service}</h2>{graphique_html}")
            
            if graphiques_html:
                html_complet = f"""
                <html>
                <head><title>Graphiques de Prévisions</title></head>
                <body>
                <h1>Graphiques de Prévisions - Rapport Santé</h1>
                {''.join(graphiques_html)}
                </body>
                </html>
                """
                
                # Créer un DataFrame avec le contenu HTML
                df_graphiques = pd.DataFrame([{'Contenu HTML': html_complet}])
                df_graphiques.to_excel(writer, sheet_name='Graphiques HTML', index=False)
            
        self.logger.info("✅ Rapport Excel amélioré généré avec succès.")
    
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
            previsions_formatted[service] = data # Le dictionnaire est déjà au bon format
        
        self.resultats_prevision = previsions_formatted
        return self.resultats_prevision