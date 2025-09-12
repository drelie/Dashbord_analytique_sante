# 🏥 Tableau de Bord d'Optimisation des Ressources de Santé  

Une solution de science des données pour **prédire la demande de soins, optimiser l’allocation du personnel** et fournir un **tableau de bord interactif** d’aide à la décision.  

---

## 🎯 Objectifs du Projet  

- Prévoir la demande de soins avec des modèles avancés.  
- Optimiser l’allocation du personnel selon cette demande.  
- Offrir un tableau de bord interactif pour la prise de décision.  
- Quantifier les économies et gains d’efficacité.  
- Exploiter des données cliniques historiques *(ex: CSU-Abatta)*.  

---

## 📊 Fonctionnalités Clés  

### 🔮 Prévision de la Demande  
- Modèles : **Prophet, ARIMA, Random Forest**.  
- Méthode par défaut : **ensemble de modèles**.  
- Évaluation : **sMAPE, MAE, R², RMSE**.  

### ⚡ Optimisation des Ressources  
- Analyse efficacité = **Consultations / Consultants**.  
- Recommandations de personnel saisonnier.  
- Analyse coût-bénéfice *(1 consultant = 1000 UM/mois)*.  
- **Simulateur interactif**.  

### 📈 Tableau de Bord Interactif  
- Interface **Streamlit multi-onglets** (Analyse, Prévisions, Optimisation & KPIs).  
- Visualisation des tendances et anomalies.  
- Export des rapports (Excel + graphiques interactifs).  

---

## 📦 Structure du Projet  

```
Project_complete/
├── run_analysis.py           # Analyse + génération résultats (.pkl)
├── tableau_bord_sante.py     # Application Streamlit
├── analytique_sante.py       # Moteur d’analyse & prévision
├── csv_transformer_script.py # Conversion Excel → CSV
├── config.py                 # Paramètres & constantes
├── requirements.txt          # Dépendances
├── README.md                 # Documentation
├── data/                     # Données (entrée & temporaires)
├── resultats/                # Rapports & visualisations
└── modeles/                  # Sauvegarde des modèles
```

---

## 🚀 Installation & Démarrage  

### Prérequis  
- **Python 3.8+**  
- **pip**  

### Installation  

```bash
# Cloner le dépôt
git clone https://github.com/drelie/Dashbord_analytique_sante.git
cd Dashbord_analytique_sante/Project_complete

# Créer un environnement virtuel
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation  

**Étape 1 – Analyse des Données**  

```bash
python run_analysis.py <fichier_excel_1.xlsx> [fichier_excel_2.xlsx ...] [année_prévision]
# Exemple
python run_analysis.py LBS_matrice_2023.xlsx LBS_matrice_2024.xlsx 2025
```
👉 Résultats enregistrés dans `data/` au format `.pkl`.  

**Étape 2 – Lancer le Tableau de Bord**  

```bash
streamlit run tableau_bord_sante.py
```
👉 Importer le fichier `.pkl` généré à l’étape 1.  

---

## 📊 Format des Données  

- **Colonne service** (type de service de santé).  
- **Colonnes mensuelles** : JANVIER, FEVRIER, ... (valeurs numériques).  
- Exemples : consultations, nombre de consultants, prénatal, pathologies (paludisme, diarrhée, etc.).  

---

## 🧮 Modèles & Hypothèses  

- **Efficacité** = consultations / consultants.  
- **Coût consultant/mois** = 1000 UM.  
- **Seuil haute efficacité** = > 1.1 × moyenne.  
- **Prévision minimale** = 12 mois.  
- Configurations :  
  - Prophet : saisonnalité annuelle (multiplicative).  
  - ARIMA : ordre (1,1,1) + fallback robuste.  
  - Random Forest : features calendaires.  

---

## 📈 Résultats Attendus  

- Gain d’efficacité : **+10 à 15%**.  
- Précision de prévision : **MAPE < 15%**.  
- Identification : **3-4 opportunités d’optimisation majeures**.  
- Livrables : rapports Excel, visualisations, données nettoyées, tableaux de bord.  

---

## 🎓 Composants du Projet de Master  

1. **Préprocessing & EDA** (nettoyage, exploration).  
2. **Prévision** (Prophet, ARIMA, RF, ensemble).  
3. **Optimisation des ressources** (allocation, coût-bénéfice).  
4. **Tableau de bord & visualisation** (Streamlit, export).  
5. **Rapport & documentation** (analyse, recommandations).  

---

## 🔐 Confidentialité & Éthique  

- Données **anonymisées**.  
- Conformité réglementaire respectée.  

---

## 👨‍💼 Équipe Projet  

- **Auteur** : Elie SOUSSOUBIE *(Pharmacien, Master Management des SI)*  
- **Superviseur** : M. Arnaud KONAN  
- **Établissement** : Lomé Business School *(LBS)*  
- **Année académique** : 2024-2025  
