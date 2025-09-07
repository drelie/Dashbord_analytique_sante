import os
import pickle
import pandas as pd
import sys
from pathlib import Path
import traceback

# Import des modules corrigés
from analytique_sante import AnalytiqueSante
from csv_transformer_script import HealthcareExcelTransformer

def run_full_analysis(input_file_path: str, output_dir: str = "data") -> None:
    """
    Exécute l'analyse complète de bout en bout et sauvegarde les résultats.
    
    Args:
        input_file_path (str): Le chemin vers le fichier Excel brut.
        output_dir (str): Le répertoire pour sauvegarder le fichier de résultats.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Étape 1: Transformation du fichier Excel en CSV standardisé
        print("="*60)
        print("🏥 ANALYSE COMPLÈTE DES DONNÉES DE SANTÉ")
        print("="*60)
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

        # Sauvegarde du CSV transformé
        temp_csv_file = os.path.join(output_dir, f"{Path(input_file_path).stem}_transformed.csv")
        if not transformer.save_transformed_data(temp_csv_file):
            print("❌ Échec de la sauvegarde du fichier CSV.")
            return
        
        # Étape 2: Analyse prédictive
        print("\nÉtape 2: Analyse prédictive des données de santé...")
        analytique = AnalytiqueSante()
        analytique.charger_donnees(temp_csv_file)
        
        if analytique.df_data is None or analytique.df_data.empty:
            print("❌ Les données d'entrée pour l'analyse sont vides ou invalides.")
            print("🔍 Problèmes possibles:")
            print("   - Le fichier CSV généré ne contient pas les bonnes colonnes")
            print("   - Aucun service reconnu dans les données")
            print("   - Format de date invalide")
            return
        
        print(f"✅ Données chargées pour l'analyse: {len(analytique.df_data)} points")
        print(f"📋 Services reconnus: {analytique.df_data['SERVICE'].unique().tolist()}")
        
        # Calcul des prévisions
        print("\n🔮 Calcul des prévisions multi-modèles...")
        previsions = analytique.calculer_previsions()

        # 🆕 NOUVEAU: Génération du rapport de validation
        print("\n📊 Génération du rapport de validation...")
        try:
            rapport_validation = analytique.generer_rapport_validation()
            if not rapport_validation.empty:
                print("RAPPORT DE VALIDATION DES MODÈLES:")
                print("-" * 80)
                # Afficher seulement les colonnes importantes
                colonnes_importantes = ['Service', 'Modèle', 'R² Original', 'R² Final', 
                                        'Overfitting détecté', 'Meilleur modèle']
                rapport_simplifie = rapport_validation[colonnes_importantes]
                print(rapport_simplifie.to_string(index=False, max_colwidth=15))
                print("-" * 80)
                
                # Sauvegarder le rapport
                rapport_path = os.path.join(output_dir, f"rapport_validation_{analytique.annee_donnees}.csv")
                rapport_validation.to_csv(rapport_path, index=False)
                print(f"✅ Rapport de validation sauvegardé: {os.path.basename(rapport_path)}")
                
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération du rapport de validation: {e}")

        # Vérifier si des prévisions ont été générées
        services_avec_previsions = [s for s, data in previsions.items() if 'error' not in data]
        services_avec_erreurs = [s for s, data in previsions.items() if 'error' in data]
        
        # Sauvegarder les résultats d'analyse
        annee = analytique.annee_donnees
        analysis_output_path = os.path.join(output_dir, f"analyse_complete_{annee}.pkl")
        with open(analysis_output_path, 'wb') as f:
            pickle.dump(previsions, f)
        print(f"✅ Résultats de l'analyse sauvegardés à : {os.path.basename(analysis_output_path)}")

        # Étape 3: Génération du rapport Excel
        if services_avec_previsions:
            print("\nÉtape 3: Génération du rapport d'analyse Excel...")
            excel_output_path = os.path.join(output_dir, f"rapport_analyse_{analytique.annee_donnees}.xlsx")
            
            try:
                analytique.resultats_prevision = previsions
                with open(excel_output_path, 'wb') as f:
                    analytique.generer_rapport_excel(f)
                print(f"✅ Rapport d'analyse généré avec succès à : {os.path.basename(excel_output_path)}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la génération du rapport Excel: {e}")
        else:
            print("\n⚠️ Aucune prévision réussie - rapport Excel non généré")

        # Résumé final
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE L'ANALYSE")
        print("="*60)
        print(f"📁 Fichiers générés dans '{output_dir}':")
        print(f"   - CSV transformé: {os.path.basename(temp_csv_file)}")
        print(f"   - Résultats analyse: {os.path.basename(analysis_output_path)}")
        if services_avec_previsions:
            print(f"   - Rapport Excel: {os.path.basename(excel_output_path)}")
        
        print(f"\n📈 Services analysés: {len(services_avec_previsions)}/{len(previsions)}")
        for service in services_avec_previsions:
            best_model_name = previsions[service]['meilleur_modele']
            score_composite = previsions[service].get('score_composite', 0)
            
            # Accès aux métriques du meilleur modèle
            best_model_metrics = previsions[service]['models'][best_model_name]['metrics']
            r2_final = best_model_metrics.get('r2', 0)
            overfitting = best_model_metrics.get('overfitting_detected', False)
            overfitting_icon = "🚨" if overfitting else "✅"
            
            print(f"   {overfitting_icon} {service}: {best_model_name} (Score: {score_composite:.3f}, R²: {r2_final:.3f})")
            
        if services_avec_erreurs:
            print(f"\n⚠️ Services avec erreurs:")
            for service in services_avec_erreurs:
                error_msg = previsions[service]['error']
                print(f"   ❌ {service}: {error_msg}")
        
        print("\n🎉 Analyse terminée avec succès!")

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE lors de l'exécution de l'analyse:")
        print(f"   {str(e)}")
        print(f"   Type d'erreur: {type(e).__name__}")
        print(f"\n🐛 Trace complète de l'erreur:")
        traceback.print_exc()

def main():
    """Fonction principale avec gestion améliorée des arguments."""
    if len(sys.argv) < 2:
        print("❌ Usage incorrect!")
        print("Usage: python run_analysis.py <chemin_vers_le_fichier_excel> [dossier_sortie]")
        print("\nExemple:")
        print("   python run_analysis.py LBS_matrice_2023.xlsx")
        print("   python run_analysis.py LBS_matrice_2023.xlsx ./resultats")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    # Vérification de l'existence du fichier d'entrée
    if not os.path.exists(input_file):
        print(f"❌ Le fichier '{input_file}' n'existe pas!")
        sys.exit(1)
    
    print(f"🔍 Fichier d'entrée: {input_file}")
    print(f"📁 Dossier de sortie: {output_dir}")
    print()
    
    run_full_analysis(input_file, output_dir)

if __name__ == "__main__":
    main()