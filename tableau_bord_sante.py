# Tableau de Bord d'Optimisation des Ressources Santé - VERSION COMPLÈTE

import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import pickle
import numpy as np
import xlsxwriter
from datetime import datetime
from io import BytesIO
from typing import Optional, Dict, Any, List
from model_cache import ModelCache

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Optimisation des Ressources Santé")

# Initialisation du gestionnaire de cache des modèles
cache_manager = ModelCache(cache_dir="cache_dir")

@st.cache_data
def load_analysis_results(uploaded_file: BytesIO) -> Optional[Dict[str, Any]]:
    """
    Charge les résultats de l'analyse depuis un fichier .pkl téléversé.
    
    Args:
        uploaded_file (BytesIO): L'objet fichier téléversé par Streamlit.
    """
    try:
        return pickle.load(uploaded_file)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'analyse : {e}")
        return None

def calculer_score_composite_display(metrics: Dict[str, Any], contexte_sanitaire: bool = True) -> float:
    """
    Version simplifiée du calcul de score composite pour l'affichage.
    """
    if 'error' in metrics:
        return -float('inf')
    
    r2 = metrics.get('r2', 0)
    mae = metrics.get('mae', float('inf'))
    rmse = metrics.get('rmse', float('inf'))
    smape = metrics.get('smape', float('inf'))
    
    if pd.isna(r2) or pd.isna(mae) or pd.isna(rmse) or pd.isna(smape):
        return -float('inf')
    
    # Normalisation des métriques
    r2_norm = max(0, min(1, r2))
    mae_norm = 1 / (1 + mae) if mae != float('inf') else 0
    rmse_norm = 1 / (1 + rmse) if rmse != float('inf') else 0
    smape_norm = 1 / (1 + smape / 100) if smape != float('inf') else 0
    
    if contexte_sanitaire:
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

@st.cache_data
def generate_excel_report(_previsions_data: Dict[str, Any]) -> BytesIO:
    """
    Génère un rapport Excel en mémoire à partir des données de prévision.
    
    Args:
        _previsions_data (Dict[str, Any]): Les données de prévisions.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Feuille 1: Résumé des Modèles avec Score Composite
        df_resume = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                
                # Structure standardisée des données
                meilleur_modele = data['meilleur_modele']['nom']
                metrics = data['meilleur_modele']['metrics']
                score_composite = data.get('score_composite', calculer_score_composite_display(metrics))
                
                df_resume = pd.concat([df_resume, pd.DataFrame([{
                    "Service": service,
                    "Meilleur Modèle": meilleur_modele,
                    "Score Composite": score_composite,
                    "R2 du Modèle": metrics.get('r2'),
                    "MAE du Modèle": metrics.get('mae'),
                    "RMSE du Modèle": metrics.get('rmse'),
                    "sMAPE du Modèle": metrics.get('smape'),
                }])], ignore_index=True)

        if not df_resume.empty:
            df_resume = df_resume.sort_values('Score Composite', ascending=False)
            df_resume.to_excel(writer, sheet_name='Résumé des Modèles', index=False)
        
        # Feuille 2: Prévisions Détaillées
        df_toutes_previsions = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                df_temp = data['meilleur_modele']['forecast'].copy()
                df_temp['Service'] = service
                df_temp['Date'] = df_temp['ds']
                df_temp['Prévision'] = df_temp['yhat']
                df_toutes_previsions = pd.concat([df_toutes_previsions, df_temp[['Date', 'Service', 'Prévision']]], ignore_index=True)
        
        if not df_toutes_previsions.empty:
            df_toutes_previsions.to_excel(writer, sheet_name='Prévisions Détaillées', index=False)
        
        # Feuille 3: Comparaison de tous les modèles avec scores
        df_evaluation = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                meilleur_modele_name = data['meilleur_modele']['nom']
                
                for model_name, model_data in data['models'].items():
                    metrics = model_data['metrics']
                    score = calculer_score_composite_display(metrics)
                    is_best = (model_name == meilleur_modele_name)
                    
                    df_evaluation = pd.concat([df_evaluation, pd.DataFrame([{
                        "Service": service,
                        "Modèle": model_name,
                        "Score Composite": score,
                        "Sélectionné": "✅" if is_best else "❌",
                        "R2 du Modèle": metrics.get('r2'),
                        "MAE du Modèle": metrics.get('mae'),
                        "RMSE du Modèle": metrics.get('rmse'),
                        "sMAPE du Modèle": metrics.get('smape'),
                    }])], ignore_index=True)
        
        if not df_evaluation.empty:
            df_evaluation = df_evaluation.sort_values(['Service', 'Score Composite'], ascending=[True, False])
            df_evaluation.to_excel(writer, sheet_name='Comparaison Modèles', index=False)
            
    return output

def main() -> None:
    """
    Fonction principale pour l'exécution de l'application Streamlit.
    """
    st.title("🩺 Tableau de Bord d'Optimisation des Ressources Santé")
    st.markdown("*Version avec Score Composite Pondéré pour la sélection des modèles*")
    
    # Sidebar avec explication du score composite
    with st.sidebar:
        st.header("1. Chargement des Résultats")
        fichier_pkl = st.file_uploader(
            "Téléverser un fichier de résultats (.pkl)", 
            type=["pkl"],
            help="Ce fichier est généré par le script run_analysis.py"
        )
        
        st.markdown("---")
        st.header("2. Paramètres d'Optimisation")
        
        cout_consultant: int = st.number_input(
            "Coût mensuel moyen d'un consultant (UM)", 
            min_value=1, value=1000, help="Coût incluant salaire, formation et équipement."
        )
        objectif_croissance: int = st.slider(
            "Objectif de croissance annuel (%)", 
            min_value=0, max_value=100, value=15
        )
        
        st.markdown("---")
        st.header("📊 Score Composite Santé")
        st.markdown("""
        **Pondération utilisée :**
        - 🎯 **sMAPE** (40%) - Précision relative symétrique
        - 📏 **MAE** (30%) - Erreur absolue  
        - 📈 **R²** (20%) - Variance expliquée
        - ⚡ **RMSE** (10%) - Pénalité outliers
        
        *Plus le score est élevé, meilleur est le modèle*
        """)
        
        st.markdown("---")
        st.header("⚡ Gestion du Cache des Modèles")
        
        # Contrôles du cache
        col1, col2 = st.columns(2)
        with col1:
            cache_enabled = st.checkbox("Activer le cache des modèles", value=True, 
                                      help="Évite la ré-entraînement des modèles")
            if st.button("📊 Statistiques du Cache"):
                cache_stats = cache_manager.get_cache_stats()
                if cache_stats:
                    st.json(cache_stats)
                else:
                    st.warning("Aucune statistique de cache disponible")
        
        with col2:
            if st.button("🧹 Nettoyer le Cache"):
                cache_manager._clear_expired_cache()
                st.success("Cache nettoyé !")
            
            if st.button("🗑️ Vider tout le Cache"):
                cache_manager.clear_all_cache()
                st.success("Cache complètement vidé !")

    previsions_data: Optional[Dict[str, Any]] = None
    
    if fichier_pkl is not None:
        try:
            previsions_data = load_analysis_results(fichier_pkl)
            st.session_state['previsions_data'] = previsions_data
            if previsions_data:
                st.success("✅ Fichier de résultats chargé avec succès !")
            else:
                st.warning("Le fichier chargé est vide ou invalide.")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier : {e}")
            return
    else:
        st.info("🔄 Veuillez charger un fichier de résultats (.pkl) dans la barre latérale pour commencer la visualisation.")
        if 'previsions_data' in st.session_state:
            del st.session_state['previsions_data']
        return

    previsions_data = st.session_state.get('previsions_data')
    
    if previsions_data:
        tabs = st.tabs(["📈 Analyse Exploratoire", "🔮 Prévisions de la Demande", "⚙️ Optimisation & KPIs"])
        
        # --- RECONSTRUCTION des données historiques pour les graphiques ---
        df_historique_complet = pd.DataFrame()
        annee_donnees: Optional[int] = None
        for service, data in previsions_data.items():
            if 'error' not in data:
                df = data['donnees_historiques'].copy()
                df['service'] = service
                df_historique_complet = pd.concat([df_historique_complet, df], ignore_index=True)
                if annee_donnees is None:
                    annee_donnees = df['ds'].dt.year.max()
        
        # --- TAB 1: Analyse Exploratoire ---
        with tabs[0]:
            st.header("📊 Analyse Exploratoire des Données")
            
            if df_historique_complet.empty:
                st.warning("Aucune donnée historique valide trouvée pour l'analyse exploratoire.")
                return
            
            services_trouves = df_historique_complet['service'].unique()
            total_enregistrements = len(df_historique_complet)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Services Clés Analysés", len(services_trouves))
            col2.metric("Total Enregistrements", total_enregistrements)
            col3.metric("Année des Données", annee_donnees)
            
            st.markdown("---")
            st.subheader("Répartition du Volume par Service")
            fig_pie = px.pie(
                df_historique_complet.groupby('service')['y'].sum().reset_index(),
                values='y', 
                names='service', 
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent', insidetextfont=dict(color='white'))
            fig_pie.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader("Tendance Mensuelle des Services Clés")
            fig_tendance = px.line(
                df_historique_complet, x='ds', y='y', color='service', markers=True,
                labels={"ds": "Date", "y": "Valeur"}
            )
            st.plotly_chart(fig_tendance, use_container_width=True)

        # --- TAB 2: Prévisions avec Score Composite ---
        with tabs[1]:
            st.header("🔮 Prévisions de la Demande avec Score Composite")
            
            # Sélection de modèle pour visualisation
            models_and_services = {}
            for service, data in previsions_data.items():
                if 'error' not in data:
                    for model_name in data['models'].keys():
                        if model_name not in models_and_services:
                            models_and_services[model_name] = []
                        models_and_services[model_name].append(service)
            
            model_choices = ["Meilleur Modèle (Score Composite)"] + sorted(list(models_and_services.keys()))
            model_choice: str = st.selectbox(
                "Sélectionnez un Modèle pour Visualisation",
                model_choices,
                index=0, 
                help="'Meilleur Modèle' utilise le modèle avec le score composite le plus élevé."
            )
            
            st.markdown("---")
            
            for service, data in previsions_data.items():
                if 'error' in data:
                    st.warning(f"⚠️ Service '{service}': {data['error']}")
                    continue
                
                df_forecast: pd.DataFrame = pd.DataFrame()
                metrics: Dict[str, Any] = {}
                model_name_display: str = ""
                score_composite: float = 0

                if model_choice == "Meilleur Modèle (Score Composite)":
                    meilleur_modele_name = data['meilleur_modele']['nom']
                    model_data = data['meilleur_modele']
                    df_forecast = model_data['forecast']
                    metrics = model_data['metrics']
                    score_composite = data.get('score_composite', calculer_score_composite_display(metrics))
                else:
                    model_data = data['models'].get(model_choice)
                    if model_data and 'metrics' in model_data:
                        model_name_display = model_choice
                        df_forecast = model_data['forecast']
                        metrics = model_data['metrics']
                        score_composite = calculer_score_composite_display(metrics)

                if not df_forecast.empty:
                    st.subheader(f"📈 Prévisions pour : {service}")
                    
                    # Correction pour gérer les valeurs None
                    r2_value = metrics.get('r2')
                    mae_value = metrics.get('mae')
                    rmse_value = metrics.get('rmse')
                    smape_value = metrics.get('smape')
                    
                    r2_value = r2_value if r2_value is not None else 0.0
                    mae_value = mae_value if mae_value is not None else 0.0
                    rmse_value = rmse_value if rmse_value is not None else 0.0
                    smape_value = smape_value if smape_value is not None else 0.0

                    # Affichage enrichi avec le score composite
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.info(
                            f"🤖 **Modèle** : {meilleur_modele_name if model_choice == 'Meilleur Modèle (Score Composite)' else model_name_display}\n\n"
                            f"**📊 Score Composite** : {score_composite:.3f}/1.000\n\n"
                            f"**📈 R²** : {r2_value:.3f} | **📏 MAE** : {mae_value:.2f} | "
                            f"**⚡ RMSE** : {rmse_value:.2f} | **🎯 sMAPE** : {smape_value:.1f}%"
                        )
                    
                    with col2:
                        # Gauge du score composite
                        if score_composite >= 0.7:
                            st.success(f"🏆 Excellent\n{score_composite:.3f}")
                        elif score_composite >= 0.5:
                            st.warning(f"👌 Bon\n{score_composite:.3f}")
                        else:
                            st.error(f"⚠️ Faible\n{score_composite:.3f}")
                    
                    # Graphique des prévisions
                    df_hist = data['donnees_historiques'].rename(columns={'ds': 'Date', 'y': 'Historique'})
                    df_forecast_plot = df_forecast.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})
                    df_combined = pd.concat([df_hist[['Date', 'Historique']], df_forecast_plot[['Date', 'Prévision']]], ignore_index=True)
                    
                    fig = px.line(
                        df_combined, x='Date', y=['Historique', 'Prévision'],
                        title=f"Prévisions pour {service} - {meilleur_modele_name if model_choice == 'Meilleur Modèle (Score Composite)' else model_name_display} (Score: {score_composite:.3f})",
                        color_discrete_map={'Historique': 'blue', 'Prévision': 'red'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

        # --- TAB 3: Optimisation & KPIs ---
        with tabs[2]:
            st.header("⚙️ Optimisation & KPIs")
            
            total_consultants = df_historique_complet[df_historique_complet['service'] == 'Nb Consultants']['y'].sum()
            total_consultations = df_historique_complet[df_historique_complet['service'] == 'Nb Consultations']['y'].sum()
            
            total_demand = df_historique_complet['y'].sum()
            consultants_recommandes = int(total_demand / 1000)
            taux_couverture = min(100, (total_consultants / consultants_recommandes) * 100) if consultants_recommandes else 0
            cout_total = consultants_recommandes * cout_consultant
            
            monthly_totals = df_historique_complet.groupby('ds')['y'].sum()
            croissance = monthly_totals.pct_change().mean()
            croissance_percent = (croissance or 0) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Taux de couverture (Approximation)", f"{taux_couverture:.1f}%")
            col2.metric("Consultants recommandés (Heuristique)", consultants_recommandes)
            col3.metric("Coût total estimé", f"{cout_total:,.0f} UM")

            st.subheader("📈 Croissance de la demande (moyenne mensuelle)")
            st.metric("Croissance", f"{croissance_percent:.1f}%")

            st.subheader("🚨 Alertes maladies prioritaires")
            seuil_paludisme = 10
            palu_data = df_historique_complet[df_historique_complet['service'] == 'Paludisme']
            if not palu_data.empty:
                palu_mean = palu_data['y'].mean()
                if palu_mean > seuil_paludisme:
                    st.error(f"⚠️ Moyenne mensuelle Paludisme élevée : {palu_mean:.1f} cas/mois (seuil : {seuil_paludisme})")
                else:
                    st.success(f"✅ Moyenne mensuelle Paludisme sous contrôle : {palu_mean:.1f} cas/mois")
            else:
                st.info("ℹ️ Les données sur le paludisme ne sont pas disponibles.")

            st.subheader("💡 Recommandations sanitaires")
            st.markdown("""
            - **Ratio consultants/population recommandé** : 1/1000
            - **Objectif de couverture vaccinale** : 95%
            - **Seuil d'alerte paludisme** : 10 cas/mois/service
            - **Score composite minimum** : 0.5 pour validation du modèle
            """)

            st.markdown("---")
            st.subheader("📊 Exporter le Rapport d'Analyse Complet")
            
            excel_data = generate_excel_report(previsions_data)

            st.download_button(
                label="📥 Télécharger le Rapport Complet (.xlsx)",
                data=excel_data,
                file_name=f"rapport_optimisation_sante_composite_{annee_donnees}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Ce rapport contient les données historiques, les prévisions avec scores composites et les KPIs d'optimisation."
            )

if __name__ == "__main__":
    main()