#!/usr/bin/env python3
"""
Healthcare Excel to CSV Transformer - VERSION 2025
=================================================

Ce script transforme les données Excel de santé (format matrice LBS) 
vers le format CSV requis par le tableau de bord analytique_sante.

Il identifie automatiquement l'année à partir du nom du fichier Excel
et la ligne d'en-tête de manière dynamique.
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path

class HealthcareExcelTransformer:
    """Transforme les données Excel de santé vers le format CSV du dashboard"""
    
    # Dictionnaire de standardisation des noms de services
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
    
    SERVICES_CLES = list(SERVICE_STANDARD.values())
    
    def __init__(self):
        self.original_data = None
        self.transformed_data = None
        self.year = None
        self.header_row_index = None
        
        # Mapping des colonnes avec toutes les variantes possibles
        self.column_mapping = {
            # Mapping Excel headers → CSV headers avec variantes
            'JANVIER': 'JANVIER',
            'JAN': 'JANVIER',
            'JANUARY': 'JANVIER',
            'JANV': 'JANVIER',
            
            'FEVRIER': 'FEVRIER',
            'FEV': 'FEVRIER',
            'FEBRUARY': 'FEVRIER',
            'FÉVRIER': 'FEVRIER',
            'FEB': 'FEVRIER',
            
            'MARS': 'MARS',
            'MAR': 'MARS',
            'MARCH': 'MARS',
            
            'T1': 'T1',
            'TRIM1': 'T1',
            'TRIMESTRE1': 'T1',
            
            'AVRIL': 'AVRIL',
            'AVR': 'AVRIL',
            'APRIL': 'AVRIL',
            'APR': 'AVRIL',
            
            'MAI': 'MAI',
            'MAY': 'MAI',
            
            'JUIN': 'JUIN',
            'JUN': 'JUIN',
            'JUNE': 'JUIN',
            
            'T2': 'T2',
            'TRIM2': 'T2',
            'TRIMESTRE2': 'T2',
            
            'S1': 'S1',
            'SEMESTRE1': 'S1',
            
            'JUILLET': 'JUILLET',
            'JUL': 'JUILLET',
            'JULY': 'JUILLET',
            
            'AOÛT': 'AOÛT',
            'AOUT': 'AOÛT',
            'AUG': 'AOÛT',
            'AUGUST': 'AOÛT',
            
            'SEPTEM': 'SEPTEMBRE',
            'SEPTEMBRE': 'SEPTEMBRE',
            'SEPT': 'SEPTEMBRE',
            'SEPTEMBER': 'SEPTEMBRE',
            'SEP': 'SEPTEMBRE',
            
            'T3': 'T3',
            'TRIM3': 'T3',
            'TRIMESTRE3': 'T3',
            'T3 TO': 'T3 TO',
            'T3.1': 'T3 TO',
            'T3_2': 'T3 TO',
            
            'OCTOB': 'OCTOBRE',
            'OCTOBRE': 'OCTOBRE',
            'OCT': 'OCTOBRE',
            'OCTOBER': 'OCTOBRE',
            
            'NOVEM': 'NOVEMBRE',
            'NOVEMBRE': 'NOVEMBRE',
            'NOV': 'NOVEMBRE',
            'NOVEMBER': 'NOVEMBRE',
            
            'DÉCEM': 'DÉCEMBRE',
            'DECEM': 'DÉCEMBRE',
            'DÉCEMBRE': 'DÉCEMBRE',
            'DECEMBRE': 'DÉCEMBRE',
            'DEC': 'DÉCEMBRE',
            'DECEMBER': 'DÉCEMBRE',
            
            'T4': 'T4',
            'TRIM4': 'T4',
            'TRIMESTRE4': 'T4',
            
            'TOTAUX': 'TOTAUX',
            'TOTAL': 'TOTAUX',
            'TOTALS': 'TOTAUX',
            'GRAND TOTAL': 'TOTAUX'
        }
        
        # Mots-clés pour la détection automatique de l'en-tête (avec toutes les variantes)
        self.header_keywords = [
            # Mois
            'JAN', 'JANVIER', 'JANUARY', 'JANV',
            'FEV', 'FEVRIER', 'FEBRUARY', 'FÉVRIER', 'FEB',
            'MARS', 'MARCH', 'MAR',
            'AVR', 'AVRIL', 'APRIL', 'APR',
            'MAI', 'MAY',
            'JUIN', 'JUNE', 'JUN',
            'JUL', 'JUILLET', 'JULY',
            'AOUT', 'AOÛT', 'AUGUST', 'AUG',
            'SEPT', 'SEPTEMBRE', 'SEPTEMBER', 'SEP',
            'OCT', 'OCTOBRE', 'OCTOBER',
            'NOV', 'NOVEMBRE', 'NOVEMBER',
            'DEC', 'DECEMBRE', 'DECEMBER', 'DÉCEMBRE',
            
            # Trimestres/Semestres
            'T1', 'TRIM1', 'TRIMESTRE1',
            'T2', 'TRIM2', 'TRIMESTRE2',
            'T3', 'TRIM3', 'TRIMESTRE3',
            'T4', 'TRIM4', 'TRIMESTRE4',
            'S1', 'SEMESTRE1',
            'S2', 'SEMESTRE2',
            
            # Totaux
            'TOTAUX', 'TOTAL', 'TOTALS', 'GRAND TOTAL',
            
            # Autres
            'PAGE', 'SERVICE', 'DESCRIPTION', 'Unnamed: 1'
        ]
        self.header_keywords_upper = [kw.upper() for kw in self.header_keywords]

    def _standardiser_nom_service(self, nom: str) -> str:
        """Standardise le nom du service selon le mapping défini."""
        nom = str(nom).strip()
        for pattern, standard in self.SERVICE_STANDARD.items():
            if re.search(pattern, nom, re.IGNORECASE):
                return standard
        return nom

    def _extract_year_from_filename(self, file_path):
        """Tente d'extraire l'année (quatre chiffres) du nom du fichier."""
        filename = Path(file_path).stem
        match = re.search(r'(\d{4})', filename)
        if match:
            return match.group(1)
        return None

    def _detect_header_row(self, file_path):
        """Détecte dynamiquement la ligne d'en-tête dans le fichier Excel."""
        print("🔍 Détection dynamique de la ligne d'en-tête...")
        temp_df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=10, engine='openpyxl')

        best_row_index = -1
        max_matches = 0

        for i in range(len(temp_df)):
            # Convertir les valeurs de la ligne en string et majuscules pour la comparaison
            row_values = [str(x).strip().upper() for x in temp_df.iloc[i].dropna().tolist()]
            
            current_matches = sum(1 for kw in self.header_keywords_upper if kw in row_values)
            
            # Bonus pour les lignes contenant les mots-clés typiques des en-têtes de données
            if ('TOTAUX' in row_values or 'TOTAL' in row_values) and \
               (('JANVIER' in row_values or 'JAN' in row_values or 'T1' in row_values)):
                current_matches += 10 # Augmenter le score pour une meilleure détection

            if current_matches > max_matches:
                max_matches = current_matches
                best_row_index = i
        
        if best_row_index != -1 and max_matches > 2:
            print(f"✅ Ligne d'en-tête détectée à l'index: {best_row_index} (Excel Row: {best_row_index + 1}) avec {max_matches} correspondances.")
            return best_row_index
        else:
            print("⚠️ Impossible de détecter automatiquement la ligne d'en-tête avec suffisamment de confiance. Essai avec l'index par défaut (ligne 2, index 1).")
            return 1 # Fallback au comportement précédent pour les fichiers 2023/2024 si la détection échoue

    def _find_service_column_name(self):
        """
        Identifie dynamiquement le nom de la colonne contenant les descriptions de service.
        Priorise 'Unnamed: 1' si elle contient des descriptions textuelles,
        sinon, cherche la colonne la plus textuelle (non-numérique, non-'PAGE').
        """
        if self.original_data is None:
            return None

        # 1. Prioriser 'Unnamed: 1' si elle existe et contient des strings descriptifs
        if 'Unnamed: 1' in self.original_data.columns:
            sample_values = self.original_data['Unnamed: 1'].dropna().head(10)
            # Vérifier si au moins une valeur contient des lettres (non purement numérique)
            if any(isinstance(x, str) and re.search(r'[a-zA-Z]', x) for x in sample_values):
                print(f"✅ Colonne de service identifiée par priorité: 'Unnamed: 1'")
                return 'Unnamed: 1'
        
        # 2. Fallback: Chercher la colonne la plus textuelle parmi toutes les colonnes
        print("🔍 Recherche de la colonne de service la plus textuelle parmi toutes les colonnes...")
        best_service_col = None
        max_text_count = -1

        for col_name in self.original_data.columns:
            if pd.isna(col_name) or str(col_name).strip() == '':
                continue
            if str(col_name).strip().upper() == 'PAGE': # La colonne PAGE contient des numéros de page, pas des services
                continue
            
            text_count = 0
            sample_col = self.original_data[col_name].dropna().head(20)
            
            for val in sample_col:
                # Si la valeur est une chaîne et contient des lettres, on la compte
                if isinstance(val, str) and re.search(r'[a-zA-Z]', val):
                    text_count += 1
                # Si c'est numérique ou NaN, on l'ignore pour cette détection
                elif pd.api.types.is_numeric_dtype(type(val)) or pd.isna(val):
                    pass
            
            if text_count > max_text_count:
                max_text_count = text_count
                best_service_col = col_name

        if best_service_col and max_text_count > 0:
            print(f"✅ Colonne de service identifiée par analyse de contenu: '{best_service_col}' (avec {max_text_count} valeurs textuelles)")
            return best_service_col
        else:
            print("❌ Impossible de trouver une colonne de service textuelle adéquate. La colonne 'service' risque d'être incorrecte.")
            # Dernier recours: si tout le reste échoue, prend la première colonne non numérique et non 'PAGE'
            for col_name in self.original_data.columns:
                if pd.isna(col_name) or str(col_name).strip() == '' or str(col_name).strip().upper() == 'PAGE':
                    continue
                # Vérifier si la colonne n'est pas purement numérique sur un échantillon
                if not all(isinstance(x, (int, float)) or pd.isna(x) for x in self.original_data[col_name].dropna().head(5)):
                    print(f"⚠️ Fallback ultime: Utilisation de la colonne '{col_name}' comme colonne de service par défaut.")
                    return col_name
            return None


    def load_original_data(self, file_path):
        """Charge le fichier Excel avec la structure spécifique et extrait l'année."""
        try:
            print(f"📖 Chargement des données Excel: {file_path}")
            
            # Tenter d'extraire l'année du nom du fichier
            self.year = self._extract_year_from_filename(file_path)
            if self.year:
                print(f"🎉 Année détectée dans le nom du fichier: {self.year}")
            else:
                print("⚠️ Aucune année détectée dans le nom du fichier. Utilisation de 'inconnu'.")
                self.year = "inconnu"

            # Détecter dynamiquement la ligne d'en-tête
            self.header_row_index = self._detect_header_row(file_path)

            # Lire le fichier Excel avec les paramètres dynamiques
            self.original_data = pd.read_excel(
                file_path, 
                sheet_name=0,
                header=self.header_row_index, # Utiliser l'index d'en-tête détecté
                engine='openpyxl'
            )
            
            print(f"✅ Données chargées avec succès!")
            print(f"   - Shape: {self.original_data.shape}")
            print(f"   - Lignes: {len(self.original_data)}")
            print(f"   - Colonnes: {len(self.original_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {str(e)}")
            return False
    
    def analyze_structure(self):
        """Analyse la structure des données Excel chargées"""
        if self.original_data is None:
            print("❌ Aucune donnée chargée. Veuillez charger les données d'abord.")
            return
            
        print("\n🔍 ANALYSE DE LA STRUCTURE EXCEL")
        print("=" * 50)
        
        # Afficher les noms de colonnes
        print("Colonnes détectées:")
        for i, col in enumerate(self.original_data.columns):
            print(f"   Colonne {i}: '{col}'")
        
        # Identifier la colonne des services en utilisant la nouvelle logique
        service_col = self._find_service_column_name()
        
        if service_col:
            print(f"\nColonne services identifiée: '{service_col}'")
            
            services = self.original_data[service_col].dropna()
            services = services[services.astype(str).str.strip() != '']
            print(f"Services avec données: {len(services)}")
            
            print("Premiers services:")
            for i, service in enumerate(services.head(10)):
                print(f"   {i+1}. {service}")
        else:
            print("\n❌ Aucune colonne de service n'a pu être identifiée.")

        print(f"\nPremières lignes:")
        print(self.original_data.head())
    
    def transform_data(self):
        """Transforme les données Excel au format CSV requis"""
        if self.original_data is None:
            print("❌ Aucune donnée chargée. Veuillez charger les données d'abord.")
            return False
            
        try:
            print("\n🔄 TRANSFORMATION DES DONNÉES")
            print("=" * 50)
            
            # Identifier la colonne des services
            service_col = self._find_service_column_name()
            
            if service_col is None:
                print("❌ Impossible d'identifier la colonne des services pour la transformation.")
                return False
                
            print(f"✅ Colonne services identifiée: '{service_col}'")
            
            transformed_rows = []
            
            for index, row in self.original_data.iterrows():
                service_name = row[service_col]
                
                # Ignorer les lignes sans nom de service, les lignes "PAGE" ou celles dont le service est numérique
                if pd.isna(service_name) or \
                   str(service_name).strip() == '' or \
                   str(service_name).strip().upper() == 'PAGE' or \
                   (isinstance(service_name, (int, float)) and not pd.isna(service_name)):
                    continue
                
                # Standardiser le nom du service
                service_std = self._standardiser_nom_service(service_name)
                transformed_row = {'service': service_std}
                
                # Nouvelle approche: mapper toutes les colonnes possibles
                for actual_df_col_name in self.original_data.columns:
                    # Ignorer la colonne service et les colonnes vides
                    if actual_df_col_name == service_col or pd.isna(actual_df_col_name) or str(actual_df_col_name).strip() == '':
                        continue
                    
                    # Trouver le nom CSV correspondant
                    csv_col_name = None
                    col_name_upper = str(actual_df_col_name).strip().upper()
                    
                    # Chercher dans le mapping
                    for excel_key, csv_value in self.column_mapping.items():
                        if excel_key.upper() == col_name_upper:
                            csv_col_name = csv_value
                            break
                    
                    # Si non trouvé, ignorer cette colonne
                    if csv_col_name is None:
                        continue
                    
                    # Traiter la valeur
                    value = row[actual_df_col_name]
                    if pd.isna(value):
                        transformed_row[csv_col_name] = 0.0
                    else:
                        try:
                            clean_value = str(value).replace(',', '').replace(' ', '').strip()
                            transformed_row[csv_col_name] = float(clean_value) if clean_value else 0.0
                        except (ValueError, TypeError):
                            transformed_row[csv_col_name] = 0.0
                
                # Ne conserver que les services clés
                if service_std in self.SERVICES_CLES:
                    transformed_rows.append(transformed_row)
            
            self.transformed_data = pd.DataFrame(transformed_rows)
            
            # === DÉBUT DE LA CORRECTION ===
            # Créer une liste de colonnes uniques et ordonnées pour le DataFrame final
            unique_expected_columns = ['service']
            for csv_col_name in self.column_mapping.values():
                if csv_col_name not in unique_expected_columns:
                    unique_expected_columns.append(csv_col_name)

            # Assurer que toutes les colonnes attendues existent dans le DataFrame, en les initialisant à 0.0 si elles manquent.
            for col_name_to_check in unique_expected_columns:
                if col_name_to_check not in self.transformed_data.columns:
                    self.transformed_data[col_name_to_check] = 0.0
            
            # Réordonner les colonnes du DataFrame pour correspondre à l'ordre unique attendu
            self.transformed_data = self.transformed_data[unique_expected_columns]
            # === FIN DE LA CORRECTION ===
            
            print(f"✅ Transformation terminée! Services clés conservés: {len(self.transformed_data)}")
            print(f"   - Colonnes: {list(self.transformed_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la transformation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_transformed_data(self):
        """Valide les données transformées"""
        if self.transformed_data is None:
            print("❌ Aucune donnée transformée disponible.")
            return False
            
        print("\n✅ VALIDATION DES DONNÉES")
        print("=" * 50)
        
        # Vérifier les services clés
        found_services = []
        for service in self.SERVICES_CLES:
            matches = self.transformed_data[
                self.transformed_data['service'] == service
            ]
            if len(matches) > 0:
                found_services.append(service)
                print(f"✅ Service clé trouvé: {service}")
            else:
                print(f"⚠️ Service clé non trouvé: {service}")
        
        # Afficher un échantillon des données
        print(f"\n📊 Échantillon des données transformées:")
        print(self.transformed_data.head())
        
        # Vérifier les types de données
        print(f"\n📋 Types de données:")
        for col in self.transformed_data.columns:
            dtype = self.transformed_data[col].dtype
            sample_val = self.transformed_data[col].iloc[0] if len(self.transformed_data) > 0 else 'N/A'
            print(f"   {col}: {dtype} (exemple: {sample_val})")
        
        # Vérifier les totaux
        if 'TOTAUX' in self.transformed_data.columns:
            total_sum = self.transformed_data['TOTAUX'].sum()
            print(f"\n📈 Somme des totaux: {total_sum:,.0f}")
        
        # La validation réussit si au moins une partie des services clés est trouvée.
        return len(found_services) >= 5 # Au moins 5 services clés trouvés

    
    def save_transformed_data(self, output_path=None):
        """Sauvegarde les données transformées au format CSV en incluant l'année."""
        if self.transformed_data is None:
            print("❌ Aucune donnée transformée à sauvegarder.")
            return False
            
        try:
            if self.year:
                output_filename = f'LBS_matrice_{self.year}_cleaned.csv'
            else:
                output_filename = 'LBS_matrice_cleaned.csv'
            
            if output_path is None:
                output_path = output_filename
            
            print(f"\n💾 Sauvegarde des données: {output_path}")
            
            self.transformed_data.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Données sauvegardées avec succès!")
            print(f"   - Fichier: {output_path}")
            print(f"   - Taille: {os.path.getsize(output_path)} bytes")
            print(f"   - Lignes: {len(self.transformed_data)}")
            print(f"   - Colonnes: {len(self.transformed_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {str(e)}")
            return False
    
    def generate_summary_report(self):
        """Génère un rapport de résumé de la transformation"""
        if self.transformed_data is None:
            print("❌ Aucune donnée transformée disponible pour le rapport.")
            return
            
        print("\n📈 RAPPORT DE TRANSFORMATION")
        print("=" * 50)
        
        total_services = len(self.transformed_data)
        
        # Rechercher les totaux des services clés
        consultants_row = self.transformed_data[
            self.transformed_data['service'].str.contains('Nb Consultants', case=False, na=False)
        ]
        consultations_row = self.transformed_data[
            self.transformed_data['service'].str.contains('Nb Consultations', case=False, na=False)
        ]
        
        total_consultants = consultants_row['TOTAUX'].sum() if not consultants_row.empty else 0
        total_consultations = consultations_row['TOTAUX'].sum() if not consultations_row.empty else 0
        
        print(f"📊 Métriques clés (Année: {self.year if self.year else 'N/A'}):")
        print(f"   - Services transformés: {total_services}")
        print(f"   - Total consultants: {total_consultants:,.0f}")
        print(f"   - Total consultations: {total_consultations:,.0f}")
        if total_consultants > 0:
            print(f"   - Consultations par consultant: {total_consultations/total_consultants:.2f}")
        
        print(f"\n✅ Données prêtes pour analytique_sante!")


def main():
    """Fonction principale d'exécution"""
    print("🏥 HEALTHCARE EXCEL TO CSV TRANSFORMER - VERSION 2025")
    print("=" * 70)
    print("Transforme les données Excel LBS vers le format CSV analytique_sante")
    print("Identifie automatiquement l'année et la ligne d'en-tête.")
    print()
    
    transformer = HealthcareExcelTransformer()
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        possible_files = [
            'LBS_matrice_2024.xlsx',
            'matrice_2024.xlsx',
            'LBS_matrice.xlsx',
            'healthcare_data.xlsx',
            'LBS_matrice_2023.xlsx',
            'LBS_matrice_2025.xlsx',
            'LBS_matrice_2020.xlsx'
        ]
        
        input_file = None
        for file in possible_files:
            if os.path.exists(file):
                input_file = file
                break
        
        if input_file is None:
            print("❌ Aucun fichier Excel trouvé. Spécifiez le chemin:")
            print("   python csv_transformer_script.py <chemin_vers_fichier.xlsx>")
            print()
            print("Ou placez un de ces fichiers dans le répertoire courant:")
            for file in possible_files:
                print(f"   - {file}")
            return
    
    if not os.path.exists(input_file):
        print(f"❌ Fichier non trouvé: {input_file}")
        return
    
    steps = [
        ("Chargement des données Excel", lambda: transformer.load_original_data(input_file)),
        ("Analyse de la structure", lambda: transformer.analyze_structure() or True),
        ("Transformation des données", lambda: transformer.transform_data()),
        ("Validation des résultats", lambda: transformer.validate_transformed_data()),
        ("Sauvegarde CSV", lambda: transformer.save_transformed_data()),
        ("Génération du rapport", lambda: transformer.generate_summary_report() or True)
    ]
    
    print("🚀 DÉMARRAGE DU PIPELINE DE TRANSFORMATION")
    print("=" * 70)
    
    for step_name, step_func in steps:
        print(f"\n⏳ {step_name}...")
        try:
            result = step_func()
            if result is False:
                print(f"❌ Échec à l'étape: {step_name}")
                return
        except Exception as e:
            print(f"❌ Erreur dans {step_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n🎉 TRANSFORMATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 70)
    print("✅ Vos données sont prêtes pour analytique_sante!")
    print(f"✅ Le fichier généré sera nommé 'LBS_matrice_{transformer.year}_cleaned.csv' (ou 'LBS_matrice_cleaned.csv' si l'année est inconnue)")
    print()
    print("Prochaines étapes:")
    print("1. Vérifiez le fichier CSV généré")
    print("2. Lancez analytique_sante avec le nouveau fichier")
    print("3. Le format est compatible avec votre tableau de bord existant")


if __name__ == "__main__":
    main()