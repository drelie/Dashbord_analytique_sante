#!/usr/bin/env python3
"""
Healthcare Excel to CSV Transformer - VERSION 2025 CORRECTED

Ce script transforme les données Excel de santé (format matrice LBS) 
vers le format CSV requis par le tableau de bord `analytique_sante`.

Il identifie automatiquement l'année à partir du nom du fichier Excel
et la ligne d'en-tête de manière dynamique.
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

class HealthcareExcelTransformer:
    """
    Transforme les données Excel de santé vers le format CSV du dashboard.
    
    Cette classe gère l'ensemble du pipeline de transformation :
    - Détection dynamique de l'année et de l'en-tête
    - Nettoyage et standardisation des noms de colonnes et de services
    - Exportation des données transformées en un fichier CSV
    """
    
    SERVICE_STANDARD: Dict[str, str] = {
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
    
    SERVICES_CLES: List[str] = list(SERVICE_STANDARD.values())
    
    def __init__(self) -> None:
        """Initialise une nouvelle instance du transformateur."""
        self.original_data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.year: Optional[str] = None
        self.header_row_index: Optional[int] = None
        # Mapping amélioré des mois
        self.month_mapping: Dict[str, str] = {
            # Français
            'JANVIER': 'January', 'JANV': 'January', 'JAN': 'January',
            'FEVRIER': 'February', 'FÉVRIER': 'February', 'FEV': 'February', 'FÉV': 'February', 'FEB': 'February',
            'MARS': 'March', 'MAR': 'March',
            'AVRIL': 'April', 'AVR': 'April', 'APR': 'April',
            'MAI': 'May',
            'JUIN': 'June', 'JUN': 'June',
            'JUILLET': 'July', 'JUIL': 'July', 'JUL': 'July',
            'AOUT': 'August', 'AOÛT': 'August', 'AOU': 'August', 'AUG': 'August',
            'SEPTEMBRE': 'September', 'SEPTEM': 'September', 'SEPT': 'September', 'SEP': 'September',
            'OCTOBRE': 'October', 'OCTOB': 'October', 'OCT': 'October',
            'NOVEMBRE': 'November', 'NOVEM': 'November', 'NOV': 'November',
            'DECEMBRE': 'December', 'DÉCEMBRE': 'December', 'DECEM': 'December', 'DEC': 'December', 'DÉCEM': 'December',
            # Anglais (au cas où)
            'JANUARY': 'January', 'FEBRUARY': 'February', 'MARCH': 'March',
            'APRIL': 'April', 'MAY': 'May', 'JUNE': 'June',
            'JULY': 'July', 'AUGUST': 'August', 'SEPTEMBER': 'September',
            'OCTOBER': 'October', 'NOVEMBER': 'November', 'DECEMBER': 'December'
        }
        self.service_col_name: str = ''
        
    def _find_header_row(self, df: pd.DataFrame) -> int:
        """
        Détecte la ligne d'en-tête contenant les noms de mois.
        
        Args:
            df (pd.DataFrame): Le DataFrame brut chargé.
            
        Returns:
            int: L'index de la ligne d'en-tête.
        """
        for i, row in df.iterrows():
            row_str = ' '.join(str(cell).upper() for cell in row.fillna(''))
            # Recherche des mots-clés de mois français
            month_keywords = ['JANVIER', 'FEVRIER', 'FÉVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN']
            matches = sum(1 for keyword in month_keywords if keyword in row_str)
            
            if matches >= 3:  # Au moins 3 mois détectés
                print(f"✅ Ligne d'en-tête détectée à l'index: {i} (Excel Row: {i+1}) avec {matches} mois détectés.")
                return i
                
        print("⚠ Aucune ligne d'en-tête valide détectée. Utilisation de la première ligne.")
        return 0
        
    def _find_service_column(self, df: pd.DataFrame) -> str:
        """
        Identifie la colonne de service.
        
        Args:
            df (pd.DataFrame): Le DataFrame avec la bonne ligne d'en-tête.
            
        Returns:
            str: Le nom de la colonne de service.
        """
        # Priorité : une colonne avec un nom explicite comme 'service'
        for col in df.columns:
            if 'service' in str(col).lower():
                print(f"✅ Colonne de service identifiée par nom: '{col}'")
                return col
        
        # Deuxième priorité : première colonne non-numérique
        for col in df.columns:
            if df[col].dtype == 'object':
                # Vérifier si cette colonne contient des services potentiels
                sample_values = df[col].dropna().astype(str).str.strip()
                non_empty_values = sample_values[sample_values != ''].head(10)
                
                if len(non_empty_values) > 0:
                    print(f"✅ Colonne de service identifiée: '{col}' (contient du texte)")
                    return col
        
        # Dernière option : la première colonne
        print(f"⚠️ Colonne de service par défaut: '{df.columns[0]}'")
        return df.columns[0]

    def _standardize_month_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les noms de mois dans les colonnes."""
        new_columns = []
        for col in df.columns:
            col_upper = str(col).strip().upper()
            # Rechercher une correspondance dans le mapping des mois
            standardized = self.month_mapping.get(col_upper, col)
            new_columns.append(standardized)
        
        df.columns = new_columns
        return df

    def load_original_data(self, file_path: Union[str, Path]) -> bool:
        """
        Charge et prépare les données brutes d'un fichier Excel.
        
        Args:
            file_path (Union[str, Path]): Le chemin vers le fichier Excel.
            
        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        try:
            print(f"📖 Chargement des données Excel: {file_path}")
            
            # Extraction de l'année du nom de fichier
            year_match = re.search(r'(\d{4})', Path(file_path).stem)
            if year_match:
                self.year = year_match.group(1)
                print(f"✅ Année détectée: {self.year}")
            else:
                self.year = '2023'  # Valeur par défaut
                print("⚠️ Année non détectée, utilisation de 2023 par défaut.")

            # Chargement temporaire pour détecter l'en-tête
            df_temp = pd.read_excel(file_path, header=None)
            print(f"📊 Dimensions du fichier: {df_temp.shape}")
            
            # Affichage des premières lignes pour debug
            print("🔍 Aperçu des premières lignes:")
            for i in range(min(5, len(df_temp))):
                print(f"  Ligne {i}: {list(df_temp.iloc[i].dropna())[:10]}")  # Première 10 colonnes non-vides
            
            self.header_row_index = self._find_header_row(df_temp)
            
            # Chargement final avec l'en-tête correcte
            self.original_data = pd.read_excel(file_path, header=self.header_row_index)
            
            # Nettoyage des noms de colonnes
            self.original_data.columns = [str(col).strip() for col in self.original_data.columns]
            
            print(f"✅ Données chargées: {len(self.original_data)} lignes, {len(self.original_data.columns)} colonnes")
            print(f"📋 Colonnes détectées: {list(self.original_data.columns)[:10]}...")  # Première 10 colonnes
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
            import traceback
            traceback.print_exc()
            return False

    def transform(self) -> Optional[pd.DataFrame]:
        """
        Transforme les données de format large (matrice) en format long (tableau).
        
        Returns:
            Optional[pd.DataFrame]: Le DataFrame transformé, ou None en cas d'échec.
        """
        if self.original_data is None:
            print("❌ Données originales non chargées.")
            return None
            
        print("🔄 Début de la transformation...")
        df = self.original_data.copy()
        
        try:
            # 1. Standardiser les noms de mois
            df = self._standardize_month_names(df)
            print(f"📋 Colonnes après standardisation: {list(df.columns)}")
            
            # 2. Identifier la colonne de service
            self.service_col_name = self._find_service_column(df)
            
            # 3. Identifier les colonnes de mois
            month_names = {'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'}
            
            value_cols = [col for col in df.columns if col in month_names]
            print(f"📅 Colonnes de mois identifiées: {value_cols}")
            
            if not value_cols:
                print("❌ Aucune colonne de mois détectée!")
                print("🔍 Colonnes disponibles:", list(df.columns))
                return pd.DataFrame()

            # 4. Nettoyer les données avant la transformation
            print("🧹 Nettoyage des données...")
            
            # Supprimer les lignes vides dans la colonne service
            df_clean = df[df[self.service_col_name].notna() & 
                         (df[self.service_col_name].astype(str).str.strip() != '')].copy()
            
            print(f"📊 Lignes après nettoyage: {len(df_clean)}")
            
            if len(df_clean) == 0:
                print("❌ Aucune donnée valide après nettoyage!")
                return pd.DataFrame()

            # Remplacer les valeurs non numériques par NaN dans les colonnes de mois
            for col in value_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            # 5. Transformation (melt)
            print("🔀 Transformation en format long...")
            df_melted = df_clean.melt(
                id_vars=[self.service_col_name], 
                value_vars=value_cols,
                var_name='month', 
                value_name='valeur'
            )
            
            print(f"📊 Données après melt: {len(df_melted)} lignes")
            
            # 6. Supprimer les valeurs nulles
            df_melted = df_melted.dropna(subset=['valeur'])
            print(f"📊 Données après suppression des NaN: {len(df_melted)} lignes")
            
            # 7. Créer la colonne date
            print("📅 Création des dates...")
            df_melted['date'] = pd.to_datetime(
                df_melted['month'] + ' 1, ' + self.year, 
                format='%B %d, %Y',
                errors='coerce'
            )
            
            # Supprimer les dates invalides
            df_melted = df_melted.dropna(subset=['date'])
            print(f"📊 Données après création des dates: {len(df_melted)} lignes")
            
            # 8. Formater et finaliser
            if len(df_melted) > 0:
                df_melted['date'] = df_melted['date'].dt.strftime('%Y-%m-%d')
                
                # Renommer et sélectionner les colonnes finales
                self.transformed_data = df_melted.rename(columns={
                    self.service_col_name: 'SERVICE'
                })[['date', 'SERVICE', 'valeur']].copy()
                
                # Filtrer les services non vides
                self.transformed_data = self.transformed_data[
                    self.transformed_data['SERVICE'].astype(str).str.strip() != ''
                ]
                
                print(f"✅ Transformation réussie: {len(self.transformed_data)} lignes")
                print("📋 Aperçu des services:")
                for service in self.transformed_data['SERVICE'].unique()[:10]:
                    count = len(self.transformed_data[self.transformed_data['SERVICE'] == service])
                    print(f"  - {service}: {count} points de données")
                
                return self.transformed_data
            else:
                print("❌ Aucune donnée valide après transformation!")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Erreur lors de la transformation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_transformed_data(self, output_path: Union[str, Path]) -> bool:
        """
        Sauvegarde le DataFrame transformé dans un fichier CSV.
        
        Args:
            output_path (Union[str, Path]): Le chemin d'enregistrement du fichier CSV.
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon.
        """
        if self.transformed_data is None or self.transformed_data.empty:
            print("❌ Aucune donnée transformée à sauvegarder.")
            return False
            
        try:
            self.transformed_data.to_csv(output_path, index=False)
            print(f"✅ Fichier CSV sauvegardé: {output_path}")
            print(f"📊 Contenu sauvegardé: {len(self.transformed_data)} lignes")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False

def main():
    """Fonction principale pour l'exécution du script en ligne de commande."""
    if len(sys.argv) < 2:
        print("Usage: python csv_transformer_script.py <fichier_excel> [fichier_csv_sortie]")
        sys.exit(1)

    input_file_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]
    else:
        output_file_path = f"{Path(input_file_path).stem}_transformed.csv"
        
    transformer = HealthcareExcelTransformer()
    
    if transformer.load_original_data(input_file_path):
        transformed_df = transformer.transform()
        if transformed_df is not None and not transformed_df.empty:
            transformer.save_transformed_data(output_file_path)
        else:
            print("❌ La transformation a échoué - aucune donnée à sauvegarder")

if __name__ == "__main__":
    main()