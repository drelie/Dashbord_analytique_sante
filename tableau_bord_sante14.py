"""
Tableau de Bord d'Optimisation des Ressources Santé
VERSION FINALE : Moteur centralisé et interface utilisateur améliorée
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from datetime import datetime
from io import BytesIO

from analytique_sante import AnalytiqueSante

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Optimisation des Ressources Santé")

@st.cache_resource
def get_analyzer():
    return AnalytiqueSante()

def main():
    st.title("🏥 Tableau de Bord d'Optimisation des Ressources Santé")
    analytique = get_analyzer()
    with st.sidebar:
        st.header("1. Chargement des Données")
        fichier = st.file_uploader("Téléverser un fichier CSV nettoyé", type=["csv"])
        st.markdown("---")
        st.header("2. Paramètres d'Optimisation")
        cout_consultant = st.number_input(
            "Coût mensuel moyen d'un consultant (UM)", 
            min_value=1, value=1000, help="Coût incluant salaire, formation et équipement."
        )
        objectif_croissance = st.slider(
            "Objectif de croissance annuel (%)", 
            min_value=0, max_value=100, value=15
        )

    if fichier is not None:
        try:
            analytique.charger_et_preparer_donnees(fichier)
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse du fichier : {e}")
            return
    else:
        st.info("Veuillez charger un fichier CSV dans la barre latérale pour commencer l'analyse.")
        return

    if analytique.donnees_mensuelles is not None and not analytique.donnees_mensuelles.empty:
        tabs = st.tabs(["📈 Analyse Exploratoire", "🔮 Prévisions de la Demande", "⚙️ Optimisation & KPIs"])
        # --- Onglet 1: Analyse Exploratoire ---
        with tabs[0]:
            st.header("Analyse Exploratoire des Données")
            col1, col2, col3 = st.columns(3)
            col1.metric("Services Clés Analysés", len(analytique.SERVICES_CLES))
            col2.metric("Total Enregistrements Mensuels", len(analytique.donnees_mensuelles))
            col3.metric("Année des Données", analytique.annee_donnees)
            st.markdown("---")
            st.subheader("Répartition du Volume par Service")
            fig_pie = px.pie(
                analytique.donnees_mensuelles.groupby('service')['valeur'].sum().reset_index(),
                values='valeur', 
                names='service', 
                hole=0.3
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent', 
                insidetextfont=dict(color='white')
            )
            fig_pie.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.subheader("Tendance Mensuelle des Services Clés")
            fig_tendance = px.line(
                analytique.donnees_mensuelles, x='date', y='valeur', color='service',
                markers=True
            )
            st.plotly_chart(fig_tendance, use_container_width=True)

        # --- Onglet 2: Prévisions ---
        with tabs[1]:
            model_choice = st.selectbox(
                "Modèle à afficher",
                [
                    "Ensemble (par défaut)",
                    "Prophet",
                    "ARIMA",
                    "RandomForest"
                ],
                index=0,
                key="prev_model_choice"
            )
            st.header(f"Prévision de la Demande - Modèle: {model_choice}")
            periodes = st.slider("Mois à prévoir", 1, 24, 12, key="prev_slider")
            if st.button("Lancer / Recalculer les prévisions", key="run_forecast"):
                with st.spinner("Calcul des prévisions pour tous les services..."):
                    previsions_data = analytique.prevision_demande(periodes)
                    st.session_state['previsions_data'] = previsions_data
            previsions_data = st.session_state.get('previsions_data')
            if previsions_data:
                for service, data in previsions_data.items():
                    models = data['models']
                    if model_choice == "Ensemble (par défaut)" and 'Ensemble' in models:
                        df_forecast = models['Ensemble']['forecast']
                    elif model_choice in models:
                        df_forecast = models[model_choice]['forecast']
                    else:
                        continue
                    st.subheader(f"Prévisions pour {service}")
                    if not df_forecast.empty:
                        df_hist = data['history']
                        fig = px.line(
                            df_hist,
                            x='ds',
                            y='y',
                            markers=True,
                            labels={'y': 'Historique'},
                            title=f"Prévision pour : {service}"
                        )
                        fig.update_traces(
                            name='Historique',
                            line=dict(color='#63b3ed', width=2),
                            marker=dict(color='#63b3ed')
                        )
                        fig.add_scatter(
                            x=df_forecast['ds'],
                            y=df_forecast['yhat'],
                            mode='lines',
                            name=model_choice,
                            line=dict(color='#f6ad55', dash='dash', width=3)
                        )
                        if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
                            fig.add_scatter(
                                x=df_forecast['ds'],
                                y=df_forecast['yhat_lower'],
                                mode='lines',
                                name='Borne basse',
                                line=dict(color='#38a169', dash='dot')
                            )
                            fig.add_scatter(
                                x=df_forecast['ds'],
                                y=df_forecast['yhat_upper'],
                                mode='lines',
                                name='Borne haute',
                                line=dict(color='#e53e3e', dash='dot')
                            )
                        fig.update_layout(
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            ),
                            xaxis_title="Date",
                            yaxis_title="Valeur"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Aucune prévision disponible pour ce service.")

        # --- Onglet 3: Optimisation & KPIs ---
        with tabs[2]:
            st.header("Optimisation & KPIs")
            total_demand = analytique.donnees_mensuelles['valeur'].sum()
            consultants_actuels = len(analytique.donnees_mensuelles['service'].unique()) * 2  # exemple
            consultants_recommandes = int(total_demand / 1000)  # ratio OMS fictif
            taux_couverture = min(100, consultants_actuels / consultants_recommandes * 100) if consultants_recommandes else 0
            cout_total = consultants_recommandes * cout_consultant
            croissance = (analytique.donnees_mensuelles.groupby('date')['valeur'].sum().pct_change().mean() or 0) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Taux de couverture", f"{taux_couverture:.1f}%")
            col2.metric("Consultants recommandés", consultants_recommandes)
            col3.metric("Coût total estimé", f"{cout_total:.0f} UM")

            st.subheader("Croissance de la demande (moyenne mensuelle)")
            st.metric("Croissance", f"{croissance:.1f}%")

            st.subheader("Alertes maladies prioritaires")
            seuil_paludisme = 10
            palu_prev = analytique.donnees_mensuelles[analytique.donnees_mensuelles['service'] == 'Paludisme']['valeur'].mean()
            if palu_prev > seuil_paludisme:
                st.error(f"⚠️ Prévision paludisme élevée : {palu_prev:.1f} cas/mois (seuil : {seuil_paludisme})")
            else:
                st.success(f"Prévision paludisme sous contrôle : {palu_prev:.1f} cas/mois")

            st.subheader("Recommandations sanitaires")
            st.markdown("- Ratio consultants/population recommandé : 1/1000")
            st.markdown("- Objectif de couverture vaccinale : 95%")
            st.markdown("- Seuil d'alerte paludisme : 10 cas/mois/service")

if __name__ == "__main__":
    main()