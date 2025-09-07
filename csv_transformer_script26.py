#!/usr/bin/env python3
"""
Healthcare Excel to CSV Transformer - VERSION 2025 CORRIGÉE
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
        r"TDR positifs": "TDR Paludisme+",
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
        self.year: Optional[int] = None
        self.service_col_name: str = ''
        self.COLONNES_MOIS = ['JANVIER', 'FEVRIER', 'FÉVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 'JUILLET', 'AOÛT', 'AOUT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE', 'DÉCEMBRE', 'DECEMBRE']
        
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
        
    def _find_service_column(self, df: pd.DataFrame) -> str:
        """
        Identifie la colonne de service.
        
        Args:
            df (pd.DataFrame): Le DataFrame avec la bonne ligne d'en-tête.
            
        Returns:
            str: Le nom de la colonne de service.
        """
        # Recherche prioritaire par mots-clés
        for col in df.columns:
            col_upper = str(col).strip().upper()
            if 'SERVICE' in col_upper or 'INDICATEUR' in col_upper:
                print(f"✅ Colonne de service identifiée par mot-clé: '{col}'")
                return col
        
        # Deuxième priorité : première colonne non-numérique
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"✅ Colonne de service identifiée: '{col}' (première colonne non-numérique)")
                return col
        
        # Dernière option : la première colonne
        print(f"⚠️ Colonne de service par défaut: '{df.columns[0]}'")
        return df.columns[0]
        
    def _standardize_month_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les noms de mois dans les colonnes."""
        new_columns = []
        for col in df.columns:
            col_upper = str(col).strip().upper()
            standardized = self.month_mapping.get(col_upper, col)
            new_columns.append(standardized)
        
        df.columns = new_columns
        return df

    def load_original_data(self, file_path: str) -> bool:
        """
        Charge le fichier Excel, en détectant l'année et l'en-tête dynamiquement.
        
        Args:
            file_path (str): Le chemin vers le fichier Excel.
            
        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        if not os.path.exists(file_path):
            print(f"❌ Erreur: Le fichier '{file_path}' n'existe pas.")
            return False
        
        try:
            # Détecter l'année à partir du nom du fichier
            match_annee = re.search(r'(\d{4})', Path(file_path).stem)
            if match_annee:
                self.year = int(match_annee.group(1))
            else:
                print("⚠️ Année non trouvée dans le nom du fichier. Année par défaut : 2024.")
                self.year = 2024
            
            # Détecter l'en-tête (ligne contenant les mois)
            df_preview = pd.read_excel(file_path, header=None, nrows=10, engine='openpyxl')
            header_row = None
            for idx, row in df_preview.iterrows():
                if any(str(col).upper().strip() in self.COLONNES_MOIS for col in row.astype(str)):
                    header_row = idx
                    break
            
            if header_row is None:
                print("❌ Erreur: En-tête de mois non trouvé dans les 10 premières lignes.")
                return False
                
            self.original_data = pd.read_excel(file_path, header=header_row, engine='openpyxl')
            print(f"✅ Fichier Excel chargé. Année détectée: {self.year}. Ligne d'en-tête: {header_row + 1}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
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
            
            # 2. Identifier la colonne de service
            self.service_col_name = self._find_service_column(df)
            
            # 3. Identifier les colonnes de mois
            month_names = {'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'}
            
            value_cols = [col for col in df.columns if col in month_names]
            
            if not value_cols:
                print("❌ Aucune colonne de mois détectée!")
                print("🔍 Colonnes disponibles:", list(df.columns))
                return None

            # 4. Nettoyer les données avant la transformation
            df_clean = df.copy()
            df_clean = df_clean[df_clean[self.service_col_name].notna() & 
                               (df_clean[self.service_col_name].astype(str).str.strip() != '')].copy()
            
            for col in value_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean[col] = df_clean[col].apply(lambda x: x if x >= 0 else np.nan)
                    
            # 5. Transformation (melt)
            df_melted = df_clean.melt(
                id_vars=[self.service_col_name], 
                value_vars=value_cols,
                var_name='month', 
                value_name='valeur'
            )
            
            # 6. Supprimer les valeurs nulles
            df_melted = df_melted.dropna(subset=['valeur'])
            
            # 7. Créer la colonne date
            df_melted['date'] = pd.to_datetime(
                df_melted['month'] + ' 1, ' + str(self.year), 
                format='%B %d, %Y',
                errors='coerce'
            )
            
            df_melted = df_melted.dropna(subset=['date'])
            
            # 8. Formater et finaliser
            if len(df_melted) > 0:
                df_melted['date'] = df_melted['date'].dt.strftime('%Y-%m-%d')
                
                self.transformed_data = df_melted.rename(columns={
                    self.service_col_name: 'SERVICE'
                })[['date', 'SERVICE', 'valeur']].copy()
                
                self.transformed_data = self.transformed_data[
                    self.transformed_data['SERVICE'].astype(str).str.strip() != ''
                ]
                
                print(f"✅ Transformation réussie: {len(self.transformed_data)} lignes")
                return self.transformed_data
            else:
                print("❌ Aucune donnée valide après transformation!")
                return None

        except Exception as e:
            print(f"❌ Erreur lors de la transformation: {e}")
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