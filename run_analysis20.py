import os
import pickle
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

# Import des modules corrigés
from analytique_sante import AnalytiqueSante
from csv_transformer_script import HealthcareExcelTransformer

def run_full_analysis(input_file_path: str, annee_de_prevision: Optional[int] = None, output_dir: str = "data") -> None:
    """
    Exécute l'analyse complète de bout en bout et sauvegarde les résultats.
    
    Args:
        input_file_path (str): Le chemin vers le fichier Excel brut.
        annee_de_prevision (int, optional): L'année pour laquelle générer les prévisions.
                                            Les données jusqu'à (année_prév - 1) seront utilisées.
        output_dir (str): Le répertoire pour sauvegarder le fichier de résultats.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Étape 1: Transformation du fichier Excel en CSV standardisé
        print("="*70)
        print("🏥 ANALYSE COMPLÈTE DES DONNÉES DE SANTÉ - VERSION SCORE COMPOSITE")
        print("="*70)
        print("Étape 1: Transformation du fichier Excel en CSV...")
        
        transformer = HealthcareExcelTransformer()
        if not transformer.load_original_data(input_file_path):
            print("❌ Échec du chargement du fichier Excel. Arrêt de l'exécution.")
            return

        csv_data = transformer.transform()
        if csv_data is None or csv_data.empty:
            print("❌ Échec de la transformation. Aucune donnée générée.")
            print("🔍 Vérifications possibles:")
            print("   - Le fichier Excel contient-il des colonnes de mois (JANVIER, FEVRIER, etc.) ?")
            print("   - Les données sont-elles dans le bon format (lignes = services, colonnes = mois) ?")
            print("   - Y a-t-il des données numériques dans les cellules ?")
            return
        
        print(f"✅ Transformation réussie - {len(csv_data)} lignes de données")
        print(f"📊 Services détectés: {len(csv_data['SERVICE'].unique())}")

        temp_csv_file = os.path.join(output_dir, f"{Path(input_file_path).stem}_transformed.csv")
        if not transformer.save_transformed_data(temp_csv_file):
            print("❌ Échec de la sauvegarde du fichier CSV.")
            return
        
        # Étape 2: Analyse prédictive avec score composite
        print("\nÉtape 2: Analyse prédictive avec score composite pondéré...")
        analytique = AnalytiqueSante()
        analytique.charger_donnees(temp_csv_file, annee_cible=annee_de_prevision)
        
        if analytique.df_data is None or analytique.df_data.empty:
            print("❌ Les données d'entrée pour l'analyse sont vides ou invalides.")
            print("🔍 Problèmes possibles:")
            print("   - Le fichier CSV généré ne contient pas les bonnes colonnes")
            print("   - Aucun service reconnu dans les données")
            print("   - Format de date invalide")
            return
        
        print(f"✅ Données chargées pour l'analyse: {len(analytique.df_data)} points")
        print(f"📋 Services reconnus: {analytique.df_data['SERVICE'].unique().tolist()}")
        
        print("\n🔬 Calcul des prévisions avec score composite pondéré...")
        print("📊 Pondération sanitaire utilisée:")
        print("   • sMAPE (Précision relative symétrique): 40%")
        print("   • MAE (Erreur absolue): 30%") 
        print("   • R² (Variance expliquée): 20%")
        print("   • RMSE (Pénalité outliers): 10%")
        print()
        
        previsions = analytique.calculer_previsions()
        
        services_avec_previsions = [s for s, data in previsions.items() if 'error' not in data]
        services_avec_erreurs = [s for s, data in previsions.items() if 'error' in data]
        
        print(f"✅ Prévisions réussies pour {len(services_avec_previsions)} services")
        if services_avec_erreurs:
            print(f"⚠️ Prévisions échouées pour {len(services_avec_erreurs)} services: {services_avec_erreurs}")
        
        annee = annee_de_prevision if annee_de_prevision else analytique.annee_donnees
        analysis_output_path = os.path.join(output_dir, f"analyse_complete_{annee}.pkl")
        with open(analysis_output_path, 'wb') as f:
            pickle.dump(previsions, f)
        print(f"✅ Résultats de l'analyse sauvegardés à : {os.path.basename(analysis_output_path)}")

        # Étape 3: Génération du rapport Excel
        if services_avec_previsions:
            print("\nÉtape 3: Génération du rapport d'analyse Excel...")
            excel_output_path = os.path.join(output_dir, f"rapport_analyse_composite_{annee}.xlsx")
            
            try:
                analytique.resultats_prevision = previsions
                with open(excel_output_path, 'wb') as f:
                    analytique.generer_rapport_excel_complet(f)
                print(f"✅ Rapport d'analyse généré avec succès à : {os.path.basename(excel_output_path)}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la génération du rapport Excel: {e}")
        else:
            print("\n⚠️ Aucune prévision réussie - rapport Excel non généré")

        # Résumé final avec scores composites
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DE L'ANALYSE AVEC SCORES COMPOSITES")
        print("="*70)
        print(f"📁 Fichiers générés dans '{output_dir}':")
        print(f"   - CSV transformé: {os.path.basename(temp_csv_file)}")
        print(f"   - Résultats analyse: {os.path.basename(analysis_output_path)}")
        if services_avec_previsions:
            print(f"   - Rapport Excel: {os.path.basename(excel_output_path)}")
        
        print(f"\n📈 Services analysés: {len(services_avec_previsions)}/{len(previsions)}")
        
        # Tri des services par score composite pour affichage
        services_scores = []
        for service in services_avec_previsions:
            score_composite = previsions[service].get('score_composite', 0)
            services_scores.append((service, score_composite))
        
        services_scores.sort(key=lambda x: x[1], reverse=True)

        print("\n📈 Classement des services par Score Composite :")
        for service, score in services_scores:
            print(f"   - {service:<20}: {score:.3f}")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE lors de l'exécution de l'analyse :")
        print(f"   {str(e)}")
        print(f"   Type d'erreur : {type(e).__name__}")
        import traceback
        print(f"\n🐛 Trace complète de l'erreur :")
        traceback.print_exc()

def main():
    """Fonction principale avec gestion améliorée des arguments."""
    if len(sys.argv) < 2:
        print("❌ Usage incorrect !")
        print("Usage: python run_analysis.py <chemin_vers_le_fichier_excel> [annee_de_prevision] [dossier_sortie]")
        print("\nExemple :")
        print("   python run_analysis.py LBS_matrice_2023.xlsx 2024")
        print("   python run_analysis.py LBS_matrice_2023.xlsx 2025 ./resultats")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    annee_de_prevision = None
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        annee_de_prevision = int(sys.argv[2])
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    else:
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        
    run_full_analysis(input_file, annee_de_prevision, output_dir)
    
if __name__ == "__main__":
    main()