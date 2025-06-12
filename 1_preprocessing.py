"""
Ce code est une version simplifiée du prétraitement pour le système de recommandation CV-offres.
Il se concentre uniquement sur les variables essentielles pour faciliter le matching via RAG.
"""
# Chargement packages
import pandas as pd
import re
import os

#*------ Chargement & Résumé des Datasets -------*#
#*------ Dataset Offres d'emploi ------*#
job_df = pd.read_csv("data/postings.csv")
print(f"Affichage des premières lignes:{job_df.head()}")
print(f"Informations sur le dataset:{job_df.info()}") 
print(f" Dataset offres d'emploi original: {job_df.shape[0]} offres, {job_df.shape[1]} colonnes")
# NA's & dupplication
print(job_df.isnull().sum()) 
print(job_df.duplicated().sum())

# Filtrage des colonnes essentielles pour la combinaison
job_df = job_df[['job_id', 'title', 'description', 'skills_desc']]

# L'intégration de toutes les lignes entraînerait des coûts de calcul excessifs
# Sélection aléatoire de 1000 lignes 
job_sampled_df = job_df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Échantillon de {len(job_sampled_df)} offres d'emploi")

# Prétraitement du texte
def preprocess_text(text):
    """Prétraite le texte en le mettant en minuscules, en supprimant les caractères spéciaux et en gérant les NaN."""
    if pd.isnull(text):
        return ""
    text = str(text).lower()  # Mettre en minuscules
    text = re.sub(r'[^\w\s]', ' ', text)  # Remplacer caractères spéciaux par espaces
    text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
    return text.strip()

# Combinaison des caractéristiques (comme dans le script original)
def combine_features(row):
    """Combine les caractéristiques pertinentes en une seule chaîne de texte."""
    features = []
    for col in ['title', 'description', 'skills_desc']:
        if not pd.isnull(row[col]):
            features.append(f"{col.capitalize()}: {preprocess_text(row[col])}\n")
    return ' '.join(features)

# Appliquer la combinaison des caractéristiques
job_sampled_df['processed_text'] = job_sampled_df.apply(combine_features, axis=1)

# Ne garder que les colonnes essentielles après prétraitement
job_sampled_df = job_sampled_df[['job_id', 'processed_text', 'title']]

#*------ Dataset CV -------*#
cv_df = pd.read_csv("data/Resume.csv")
print(f"Affichage des premières lignes:{cv_df.head()}")
print(f"Informations sur le dataset:{cv_df.info()}") 
print(f"Dataset CV original: {cv_df.shape[0]} CV, {cv_df.shape[1]} colonnes")
# NA's & dupplication
print(cv_df.isnull().sum()) 
print(cv_df.duplicated().sum())

# Renommage de l'ID
cv_df = cv_df.rename(columns={'ID': 'cv_id'})

# Sélection aléatoire de 1000 lignes
cv_sampled_df = cv_df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Échantillon de {len(cv_sampled_df)} CV")

# Prétraitement des CV
cv_sampled_df['processed_text'] = cv_sampled_df['Resume_str'].apply(preprocess_text)

# Filtrage des colonnes essentielles après prétraitement
cv_sampled_df = cv_sampled_df[['cv_id', 'processed_text', 'Category']]

# Vérification des données prétraitées
print(f"Offres: {len(job_sampled_df)} avec {job_sampled_df['processed_text'].isnull().sum()} valeurs manquantes")
print(f"CV: {len(cv_sampled_df)} avec {cv_sampled_df['processed_text'].isnull().sum()} valeurs manquantes")

# Afficher un exemple pour chaque type
print(job_sampled_df['processed_text'].iloc[0][:200] + "...")
print(cv_sampled_df['processed_text'].iloc[0][:200] + "...")

#*------ Export des données prétraitées ------*#
output_job_path = 'data/processed_jobs_v2.csv'
output_cv_path = 'data/processed_resumes_v2.csv'

job_sampled_df.to_csv(output_job_path, index=False)
cv_sampled_df.to_csv(output_cv_path, index=False)
