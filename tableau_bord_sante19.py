# Tableau de Bord d'Optimisation des Ressources Santé - VERSION CORRIGÉE
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

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Optimisation des Ressources Santé")

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

@st.cache_data
def generate_excel_report(_previsions_data: Dict[str, Any]) -> BytesIO:
    """
    Génère un rapport Excel en mémoire à partir des données de prévision.
    
    Args:
        _previsions_data (Dict[str, Any]): Les données de prévisions.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Feuille 1: Résumé des Modèles (avec toutes les métriques)
        df_resume = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                meilleur_modele = data['meilleur_modele']
                metrics = data['models'][meilleur_modele]['metrics']
                
                df_resume = pd.concat([df_resume, pd.DataFrame([{
                    "Service": service,
                    "Meilleur Modèle": meilleur_modele,
                    "R2 du Modèle": metrics.get('r2'),
                    "MAE du Modèle": metrics.get('mae'),
                    "RMSE du Modèle": metrics.get('rmse'),
                    "MAPE du Modèle": metrics.get('mape'),
                }])], ignore_index=True)

        if not df_resume.empty:
            df_resume.to_excel(writer, sheet_name='Résumé des Modèles', index=False)
        
        # Feuille 2: Prévisions Détaillées
        df_toutes_previsions = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                df_temp = data['models'][data['meilleur_modele']]['forecast'].copy()
                df_temp['Service'] = service
                df_temp['Date'] = df_temp['ds']
                df_temp['Prévision'] = df_temp['yhat']
                df_toutes_previsions = pd.concat([df_toutes_previsions, df_temp[['Date', 'Service', 'Prévision']]], ignore_index=True)
        
        if not df_toutes_previsions.empty:
            df_toutes_previsions.to_excel(writer, sheet_name='Prévisions Détaillées', index=False)
        
        # Feuille 3: Métriques d'Évaluation (détails de tous les modèles testés)
        df_evaluation = pd.DataFrame()
        for service, data in _previsions_data.items():
            if 'error' not in data:
                for model_name, model_data in data['models'].items():
                    metrics = model_data['metrics']
                    df_evaluation = pd.concat([df_evaluation, pd.DataFrame([{
                        "Service": service,
                        "Modèle": model_name,
                        "R2 du Modèle": metrics.get('r2'),
                        "MAE du Modèle": metrics.get('mae'),
                        "RMSE du Modèle": metrics.get('rmse'),
                        "MAPE du Modèle": metrics.get('mape'),
                    }])], ignore_index=True)
        
        if not df_evaluation.empty:
            df_evaluation.to_excel(writer, sheet_name='Evaluation des Modèles', index=False)
            
    return output

def main() -> None:
    """
    Fonction principale pour l'exécution de l'application Streamlit.
    """
    st.title("🏥 Tableau de Bord d'Optimisation des Ressources Santé")
    
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
        st.info("Veuillez charger un fichier de résultats (.pkl) dans la barre latérale pour commencer la visualisation.")
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
                    annee_donnees = df['ds'].dt.year.max() + 1 # L'année de prévision est l'année suivant les données historiques
        
        # --- TAB 1: Analyse Exploratoire ---
        with tabs[0]:
            st.header("Analyse Exploratoire des Données")
            
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

        # --- TAB 2: Prévisions de la Demande ---
        with tabs[1]:
            st.header("Prévision de la Demande")
            
            models_and_services = {}
            for service, data in previsions_data.items():
                if 'error' not in data:
                    for model_name in data['models'].keys():
                        if model_name not in models_and_services:
                            models_and_services[model_name] = []
                        models_and_services[model_name].append(service)
            
            model_choices = ["Meilleur Modèle (Auto)"] + sorted(list(models_and_services.keys()))
            model_choice: str = st.selectbox(
                "Sélectionnez un Modèle",
                model_choices,
                index=0, 
                help="Sélectionnez 'Meilleur Modèle' pour voir la prévision la plus performante (basée sur R²)."
            )
            
            st.markdown("---")
            
            for service, data in previsions_data.items():
                if 'error' in data:
                    st.warning(f"⚠️ Service '{service}': {data['error']}")
                    continue
                
                df_forecast: pd.DataFrame = pd.DataFrame()
                metrics: Dict[str, Any] = {}
                model_name_display: str = ""

                if model_choice == "Meilleur Modèle (Auto)":
                    meilleur_modele_name = data.get('meilleur_modele')
                    if meilleur_modele_name:
                        model_name_display = meilleur_modele_name
                        model_data = data['models'].get(meilleur_modele_name)
                        if model_data and 'metrics' in model_data:
                            df_forecast = model_data['forecast']
                            metrics = model_data['metrics']
                else:
                    model_data = data['models'].get(model_choice)
                    if model_data and 'metrics' in model_data:
                        model_name_display = model_choice
                        df_forecast = model_data['forecast']
                        metrics = model_data['metrics']

                if not df_forecast.empty:
                    st.subheader(f"Prévisions pour le service : {service}")
                    
                    r2_value = metrics.get('r2')
                    mae_value = metrics.get('mae')
                    rmse_value = metrics.get('rmse')
                    mape_value = metrics.get('mape')

                    st.info(
                        f"Modèle utilisé '{model_name_display}'\n"
                        f"**R²**: {r2_value:.2f} | "
                        f"**MAE**: {mae_value:.2f} | "
                        f"**RMSE**: {rmse_value:.2f} | "
                        f"**MAPE**: {mape_value:.2f}%"
                    )
                    
                    df_hist = data['donnees_historiques'].rename(columns={'ds': 'Date', 'y': 'Historique'})
                    df_forecast_plot = df_forecast.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})
                    df_combined = pd.concat([df_hist[['Date', 'Historique']], df_forecast_plot[['Date', 'Prévision']]], ignore_index=True)
                    
                    fig = px.line(
                        df_combined, x='Date', y=['Historique', 'Prévision'],
                        title=f"Prévisions de la demande pour {service} avec {model_name_display}",
                        color_discrete_map={'Historique': 'blue', 'Prévision': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # --- TAB 3: Optimisation & KPIs ---
        with tabs[2]:
            st.header("Optimisation & KPIs")
            
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

            st.subheader("Croissance de la demande (moyenne mensuelle)")
            st.metric("Croissance", f"{croissance_percent:.1f}%")

            st.subheader("Alertes maladies prioritaires")
            seuil_paludisme = 10
            palu_data = df_historique_complet[df_historique_complet['service'] == 'Paludisme']
            if not palu_data.empty:
                palu_mean = palu_data['y'].mean()
                if palu_mean > seuil_paludisme:
                    st.error(f"⚠️ Moyenne mensuelle Paludisme élevée : {palu_mean:.1f} cas/mois (seuil : {seuil_paludisme})")
                else:
                    st.success(f"Moyenne mensuelle Paludisme sous contrôle : {palu_mean:.1f} cas/mois")
            else:
                st.info("Les données sur le paludisme ne sont pas disponibles.")

            st.subheader("Recommandations sanitaires")
            st.markdown("- Ratio consultants/population recommandé : 1/1000")
            st.markdown("- Objectif de couverture vaccinale : 95%")
            st.markdown("- Seuil d'alerte paludisme : 10 cas/mois/service")

            st.markdown("---")
            st.subheader("📊 Exporter le Rapport d'Analyse Complet")
            
            excel_data = generate_excel_report(previsions_data)

            st.download_button(
                label="📥 Télécharger le Rapport Complet (.xlsx)",
                data=excel_data,
                file_name=f"rapport_optimisation_sante_{annee_donnees}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Ce rapport contient les données historiques, les prévisions de chaque modèle et les KPIs d'optimisation."
            )

if __name__ == "__main__":
    main()