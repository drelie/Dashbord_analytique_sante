# FileName: MultipleFiles/run_analysis.py
import os
import pickle
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from io import BytesIO

# Import des modules corrigés
from analytique_sante import AnalytiqueSante
from csv_transformer_script import HealthcareExcelTransformer # Assurez-vous que ce script est bien nommé ainsi

def run_full_analysis(input_file_paths: Union[str, List[str]], annee_de_prevision: Optional[int] = None, output_dir: str = "data") -> None:
    """
    Exécute l'analyse complète de bout en bout et sauvegarde les résultats.
    VERSION CORRIGÉE pour compatibilité avec analytique_sante.py amélioré.
    Prend désormais un ou plusieurs chemins de fichiers Excel en entrée.

    Args:
        input_file_paths (Union[str, List[str]]): Le chemin vers un fichier Excel brut ou une liste de chemins.
        annee_de_prevision (int, optional): L'année pour laquelle générer les prévisions.
                                            Les données jusqu'à (année_prév - 1) seront utilisées.
        output_dir (str): Le répertoire pour sauvegarder le fichier de résultats.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Convertir input_file_paths en liste si c'est une chaîne unique
        if isinstance(input_file_paths, str):
            input_file_paths = [input_file_paths]

        print("="*70)
        print("🏥 ANALYSE COMPLÈTE DES DONNÉES DE SANTÉ - VERSION SCORE COMPOSITE AMÉLIORÉE")
        print("="*70)
        print("Étape 1: Transformation des fichiers Excel en un CSV consolidé...")

        all_transformed_data = []
        transformer = HealthcareExcelTransformer()

        for i, input_file_path in enumerate(input_file_paths):
            print(f"\nTraitement du fichier Excel ({i+1}/{len(input_file_paths)}): {os.path.basename(input_file_path)}")
            if not transformer.load_original_data(input_file_path):
                print(f"⛔ Échec du chargement du fichier Excel {os.path.basename(input_file_path)}. Ce fichier sera ignoré.")
                continue

            csv_data_part = transformer.transform()
            if csv_data_part is None or csv_data_part.empty:
                print(f"⛔ Échec de la transformation pour {os.path.basename(input_file_path)}. Aucune donnée générée pour ce fichier.")
                print("🔍 Vérifications possibles:")
                print("   - Le fichier Excel contient-il des colonnes de mois (JANVIER, FEVRIER, etc.) ?")
                print("   - Les données sont-elles dans le bon format (lignes = services, colonnes = mois) ?")
                print("   - Y a-t-il des données numériques dans les cellules ?")
                continue
            
            # Assurez-vous que la colonne 'date' est bien un datetime pour le tri futur
            csv_data_part['date'] = pd.to_datetime(csv_data_part['date'])
            all_transformed_data.append(csv_data_part)
            print(f"✅ Transformation réussie pour {os.path.basename(input_file_path)} - {len(csv_data_part)} lignes de données")

        if not all_transformed_data:
            print("⛔ Aucune donnée valide n'a pu être transformée à partir des fichiers Excel fournis. Arrêt de l'exécution.")
            return

        # Concaténer toutes les données transformées
        consolidated_csv_data = pd.concat(all_transformed_data, ignore_index=True)
        # Trier par date pour s'assurer que les données sont dans le bon ordre chronologique
        consolidated_csv_data = consolidated_csv_data.sort_values(by='date').reset_index(drop=True)

        print(f"\n✅ Consolidation réussie - Total de {len(consolidated_csv_data)} lignes de données sur toutes les années.")
        print(f"📊 Services détectés dans l'ensemble consolidé: {len(consolidated_csv_data['SERVICE'].unique())}")

        # Sauvegarder le fichier CSV consolidé temporaire
        temp_csv_file = os.path.join(output_dir, f"consolidated_transformed_data.csv")
        consolidated_csv_data.to_csv(temp_csv_file, index=False)
        print(f"✅ Fichier CSV consolidé temporaire sauvegardé à : {os.path.basename(temp_csv_file)}")

        # Étape 2: Analyse prédictive avec score composite amélioré
        print("\nÉtape 2: Analyse prédictive avec score composite amélioré...")
        analytique = AnalytiqueSante()
        # Charger les données consolidées
        analytique.charger_donnees(temp_csv_file, annee_cible=annee_de_prevision)

        if analytique.df_data is None or analytique.df_data.empty:
            print("⛔ Les données d'entrée consolidées pour l'analyse sont vides ou invalides.")
            print("🔍 Problèmes possibles:")
            print("   - Le fichier CSV consolidé ne contient pas les bonnes colonnes")
            print("   - Aucun service reconnu dans les données consolidées")
            print("   - Format de date invalide")
            return

        print(f"✅ Données consolidées chargées pour l'analyse: {len(analytique.df_data)} points")
        print(f"📋 Services reconnus: {analytique.df_data['SERVICE'].unique().tolist()}")

        print("\n🔬 Calcul des prévisions avec score composite pondéré amélioré...")
        print("📊 Pondération sanitaire utilisée:")
        print("   • sMAPE (Précision relative symétrique): 40%")
        print("   • MAE (Erreur absolue): 30%")
        print("   • R² (Variance expliquée): 20%")
        print("   • RMSE (Pénalité outliers): 10%")
        print("\n🎯 Score composite borné entre 0.000 et 1.000")
        print("🔄 ARIMA robuste avec ordres multiples: (1,1,1), (0,1,1), (2,1,1), (1,1,0), (0,1,0)")
        print()

        previsions = analytique.calculer_previsions()

        services_avec_previsions = [s for s, data in previsions.items() if 'error' not in data]
        services_avec_erreurs = [s for s, data in previsions.items() if 'error' in data]

        print(f"✅ Prévisions réussies pour {len(services_avec_previsions)} services")
        if services_avec_erreurs:
            print(f"⚠️ Prévisions échouées pour {len(services_avec_erreurs)} services: {services_avec_erreurs}")

        # Déterminer l'année pour le nom du fichier de sortie
        # Si annee_de_prevision est fourni, l'utiliser. Sinon, utiliser la dernière année des données.
        if annee_de_prevision:
            output_year_str = str(annee_de_prevision)
        elif not consolidated_csv_data.empty:
            output_year_str = str(consolidated_csv_data['date'].dt.year.max())
        else:
            output_year_str = "unknown" # Fallback si aucune donnée

        analysis_output_path = os.path.join(output_dir, f"analyse_complete_{output_year_str}.pkl")
        with open(analysis_output_path, 'wb') as f:
            pickle.dump(previsions, f)
        print(f"✅ Résultats de l'analyse sauvegardés à : {os.path.basename(analysis_output_path)}")

        # Étape 3: Génération du rapport Excel avec visualisations
        if services_avec_previsions:
            print("\nÉtape 3: Génération du rapport d'analyse Excel avec visualisations...")
            excel_output_path = os.path.join(output_dir, f"rapport_analyse_ameliore_{output_year_str}.xlsx")

            try:
                analytique.resultats_prevision = previsions
                with open(excel_output_path, 'wb') as f:
                    analytique.generer_rapport_excel_complet(f)
                print(f"✅ Rapport d'analyse amélioré généré avec succès à : {os.path.basename(excel_output_path)}")
                print("📈 Nouvelles fonctionnalités du rapport:")
                print("   • Graphiques interactifs intégrés (Plotly)")
                print("   • 4 feuilles : Résumé, Comparaison, Prévisions, Graphiques HTML")
                print("   • Métriques de qualité complètes")
            except Exception as e:
                print(f"⚠️ Erreur lors de la génération du rapport Excel: {e}")
        else:
            print("\n⚠️ Aucune prévision réussie - rapport Excel non généré")

        # Résumé final avec scores composites améliorés
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DE L'ANALYSE AVEC SCORES COMPOSITES AMÉLIORÉS")
        print("="*70)
        print(f"🗂️ Fichiers générés dans '{output_dir}':")
        print(f"   - CSV consolidé temporaire: {os.path.basename(temp_csv_file)}")
        print(f"   - Résultats analyse: {os.path.basename(analysis_output_path)}")
        if services_avec_previsions:
            print(f"   - Rapport Excel amélioré: {os.path.basename(excel_output_path)}")

        print(f"\n📈 Services analysés: {len(services_avec_previsions)}/{len(previsions)}")

        # Tri des services par score composite pour affichage (scores maintenant 0-1)
        services_scores = []
        for service in services_avec_previsions:
            score_composite = previsions[service].get('score_composite', 0.0)
            services_scores.append((service, score_composite))

        services_scores.sort(key=lambda x: x[1], reverse=True)

        print("\n🏆 Classement des services par Score Composite (0-1) :")
        for service, score in services_scores:
            # Classification qualitative du score
            if score >= 0.8:
                qualite = "🟢 Excellent"
            elif score >= 0.6:
                qualite = "🔵 Bon"
            elif score >= 0.4:
                qualite = "🟡 Moyen"
            else:
                qualite = "🔴 Faible"

            print(f"   - {service:<20}: {score:.3f}/1.000 ({qualite})")

        # Statistiques des scores
        if services_scores:
            scores = [score for _, score in services_scores]
            print(f"\n📊 Statistiques des scores:")
            print(f"   • Score moyen: {np.mean(scores):.3f}/1.000")
            print(f"   • Score médian: {np.median(scores):.3f}/1.000")
            print(f"   • Meilleur score: {np.max(scores):.3f}/1.000")
            print(f"   • Score le plus faible: {np.min(scores):.3f}/1.000")

        print("="*70)

    except Exception as e:
        print(f"\n⛔ ERREUR CRITIQUE lors de l'exécution de l'analyse :")
        print(f"   {str(e)}")
        print(f"   Type d'erreur : {type(e).__name__}")
        import traceback
        print(f"\n🐛 Trace complète de l'erreur :")
        traceback.print_exc()

def main():
    """Fonction principale avec gestion améliorée des arguments."""
    if len(sys.argv) < 2:
        print("⛔ Usage incorrect !")
        print("Usage: python run_analysis.py <chemin_vers_le_fichier_excel_1> [chemin_vers_le_fichier_excel_2 ...] [annee_de_prevision] [dossier_sortie]")
        print("\nExemple :")
        print("   python run_analysis.py LBS_matrice_2023.xlsx LBS_matrice_2024.xlsx LBS_matrice_2025.xlsx 2026")
        print("   python run_analysis.py LBS_matrice_2023.xlsx LBS_matrice_2024.xlsx LBS_matrice_2025.xlsx 2026 ./resultats")
        print("\n📋 Prérequis (installer si manquant) :")
        print("   pip install pandas numpy scikit-learn statsmodels prophet xlsxwriter plotly")
        sys.exit(1)

    # Les chemins des fichiers Excel sont les premiers arguments
    input_files = []
    annee_de_prevision = None
    output_dir = "data"

    # Parcourir les arguments pour identifier les fichiers, l'année et le dossier de sortie
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.endswith(('.xlsx', '.xls')): # Si c'est un fichier Excel
            input_files.append(arg)
        elif arg.isdigit(): # Si c'est un nombre, on suppose que c'est l'année de prévision
            annee_de_prevision = int(arg)
        else: # Sinon, on suppose que c'est le dossier de sortie
            output_dir = arg
        i += 1

    if not input_files:
        print("⛔ Aucun fichier Excel d'entrée spécifié. Veuillez fournir au moins un fichier Excel.")
        sys.exit(1)

    run_full_analysis(input_files, annee_de_prevision, output_dir)

if __name__ == "__main__":
    import numpy as np # Assurez-vous que numpy est importé si utilisé dans main
    main()

