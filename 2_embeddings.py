"""
Embeddings et Matching pour CV et Offres d'Emploi (Version 2)
Processus:
1. Charger les données prétraitées v2
2. Générer des embeddings avec SentenceTransformers
3. Calculer les similarités cosinus
4. Identifier les meilleures correspondances
"""
# Chargement packages
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

#*------ Configuration du modèle d'embedding ------*#
model_name = 'all-MiniLM-L6-v2'  # Modèle léger et rapide avec de bonnes performances
model = SentenceTransformer(model_name)

#*------ Chargement des fichiers prétraités & extraction des textes ------*#
job_df = pd.read_csv("data/processed_jobs.csv")
resume_df = pd.read_csv("data/processed_resumes.csv")
print(f"\n→ Données prétraitées chargées: {len(job_df)} offres et {len(resume_df)} CV")

# Extraction des textes pour l'embedding 
job_texts = job_df['processed_text'].fillna('').tolist()
resume_texts = resume_df['processed_text'].fillna('').tolist()

#*------ Vectorisation - génération d'embeddings ------*#
job_embeddings = model.encode(job_texts, batch_size=32, show_progress_bar=True)
resume_embeddings = model.encode(resume_texts, batch_size=32, show_progress_bar=True)

print(f"→ Dimensions des embeddings d'offres: {job_embeddings.shape}")
print(f"→ Dimensions des embeddings de CV: {resume_embeddings.shape}")

# Intégration des embeddings dans les DataFrames
job_df['embedding'] = job_embeddings.tolist()
resume_df['embedding'] = resume_embeddings.tolist()

#*------ Sauvegarde des fichiers en format .pkl ------*#
os.makedirs("data/embeddings", exist_ok=True)

job_df.to_pickle("data/embeddings/job_df_with_embeddings.pkl")
resume_df.to_pickle("data/embeddings/resume_df_with_embeddings.pkl")



#*------ Exemples de Fonctions de recherche de similarité ------*#
def find_similar_cvs(job_embedding, resume_df, top_n=5):
    """Trouve les CV les plus similaires à une offre d'emploi donnée."""
    similarities = cosine_similarity([job_embedding], resume_df['embedding'].tolist())[0]
    top_indices = similarities.argsort()[-top_n:][::-1]  # Indices des meilleurs matches
    return top_indices, similarities[top_indices]

def find_similar_jobs(resume_embedding, job_df, top_n=5):
    """Trouve les offres d'emploi les plus similaires à un CV donné."""
    similarities = cosine_similarity([resume_embedding], job_df['embedding'].tolist())[0]
    top_indices = similarities.argsort()[-top_n:][::-1]  # Indices des meilleurs matches
    return top_indices, similarities[top_indices]

# Exemple 1: Trouver les meilleurs CV pour la première offre d'emploi
job_idx = 0  # Prend la première offre d'emploi
job = job_df.iloc[job_idx]
job_embedding = job['embedding']

print(f"\n→ Exemple 1: CV similaires à l'offre #{job_idx} ('{job['title'][:50]}...')")
top_cv_indices, similarities = find_similar_cvs(job_embedding, resume_df)
for i, (idx, sim) in enumerate(zip(top_cv_indices, similarities)):
    cv = resume_df.iloc[idx]
    print(f"  CV #{i+1} - ID: {cv['cv_id']} - Catégorie: {cv['Category']} - Similarité: {sim:.4f}")
    
# Exemple 2: Trouver les meilleures offres pour le premier CV
resume_idx = 0  # Prend le premier CV
cv = resume_df.iloc[resume_idx]
resume_embedding = cv['embedding']

print(f"\n→ Exemple 2: Offres similaires au CV #{resume_idx} (ID: {cv['cv_id']})")
top_job_indices, similarities = find_similar_jobs(resume_embedding, job_df)
for i, (idx, sim) in enumerate(zip(top_job_indices, similarities)):
    job = job_df.iloc[idx]
    print(f"  Offre #{i+1} - ID: {job['job_id']} - Titre: {job['title'][:50]} - Similarité: {sim:.4f}")
