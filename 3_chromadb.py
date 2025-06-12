#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChromaDB Retrieval pour CV-Job Matching - Version 2
----------------------------------------
Étape 3 du pipeline RAG CV-Job Matching:
- Charge les DataFrames enrichies avec embeddings v2
- Crée et remplit les collections ChromaDB avec les identifiants appropriés
- Recherche vectorielle entre CV et offres d'emploi
"""
# Chargement packages
import os
import pandas as pd
import chromadb

# Config
DATA_DIR = "data/embeddings"
PERSIST_DIR = "data/chromadb"  # Réutilisation du répertoire existant
os.makedirs(PERSIST_DIR, exist_ok=True)
BATCH_SIZE = 100

#*------ Chargement des données avec embeddings ------*#
job_df = pd.read_pickle(os.path.join(DATA_DIR, "job_df_with_embeddings.pkl"))
resume_df = pd.read_pickle(os.path.join(DATA_DIR, "resume_df_with_embeddings.pkl"))
print(f"→ {len(job_df)} offres | {len(resume_df)} CV chargés avec embeddings")

#*------ Initialisation ChromaDB ------*#
client = chromadb.PersistentClient(path=PERSIST_DIR)

def create_or_get_collection(name: str):
    """
    Récupère une collection existante ou en crée une nouvelle si elle n'existe pas.
    Args:
        name (str): Le nom de la collection à récupérer ou à créer.
    Returns:
        La collection ChromaDB correspondant au nom spécifié.
    """
    existing = {c.name: c for c in client.list_collections()}
    if name in existing:
        print(f"→ Collection '{name}' trouvée ({existing[name].count()} documents)")
        return client.get_collection(name)
    print(f"→ Création de la collection '{name}'")
    return client.create_collection(name)

def populate_collection(collection, df, id_prefix, text_col, id_col, metadata_fields):
    """Remplit une collection ChromaDB avec les embeddings et métadonnées d'un DataFrame.
    
    Args:
        collection: La collection ChromaDB à remplir
        df: Le DataFrame contenant les embeddings
        id_prefix: Préfixe pour les IDs des documents (ex: 'cv' ou 'job')
        text_col: Nom de la colonne contenant le texte principal
        id_col: Nom de la colonne contenant l'identifiant unique
        metadata_fields: Liste des colonnes à utiliser comme métadonnées
    """
    if collection.count() > 0:
        print(f"Collection '{collection.name}' déjà remplie ({collection.count()} documents)")
        return

    print(f"Insertion dans '{collection.name}'...")
    ids = [f"{id_prefix}_{i}" for i in df.index]
    embeddings = df['embedding'].tolist()
    documents = df[text_col].fillna('').astype(str).tolist()
    
    # Création des métadonnées avec l'ID comme information principale
    metadatas = []
    for i, idx in enumerate(df.index):
        record = {}
        # Ajouter l'ID unique comme métadonnée principale
        record["id"] = str(df.at[idx, id_col])
        # Ajouter les autres métadonnées
        for field in metadata_fields:
            if field in df.columns:
                record[field] = str(df.at[idx, field])
        metadatas.append(record)

    for i in range(0, len(df), BATCH_SIZE):
        end_idx = min(i+BATCH_SIZE, len(df))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"  → Lot {i//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1} ajouté")

#*------ Préparation des collections & insertion ------*#
# Création des collections
cv_collection = create_or_get_collection("cv_collection")
job_collection = create_or_get_collection("job_collection")

# Remplissage des collections
populate_collection(
    cv_collection,     # Collection ChromaDB cible
    resume_df,         # DataFrame contenant les données des CV
    "cv",              # Préfixe pour les IDs internes de ChromaDB ("cv_0", "cv_1", etc.)
    "processed_text",  # Colonne du DataFrame contenant le texte à stocker comme document
    "cv_id",           # Colonne du DataFrame contenant l'ID unique qui sera stocké dans metadata["id"]
    ["Category"]       # Colonnes supplémentaires à inclure dans les métadonnées
)

# Remplissage de la collection Offres d'emploi
print("\n→ Remplissage de la collection Offres d'emploi...")
populate_collection(
    job_collection,    # Collection ChromaDB cible
    job_df,            # DataFrame contenant les données des offres
    "job",             # Préfixe pour les IDs internes de ChromaDB ("job_0", "job_1", etc.)
    "processed_text",  # Colonne du DataFrame contenant le texte à stocker comme document
    "job_id",          # Colonne du DataFrame contenant l'ID unique qui sera stocké dans metadata["id"]
    ["title"]          # Colonnes supplémentaires à inclure dans les métadonnées
)

print(f"Collection CV: {cv_collection.count()} documents")    
print(f"Collection Job: {job_collection.count()} documents")  

#*------ Exemples de requêtes ------*#
print("\n🔹 Exemples de requêtes...")

# Exemple 1: Récupérer un CV par ID
first_cv = cv_collection.get(
    ids=["cv_0"],
    include=["documents", "metadatas"]
)
if first_cv and first_cv['documents']:
    print("\nExemple de CV récupéré:")
    print(f"Document: {first_cv['documents'][0][:150]}...")
    print(f"Métadonnées: {first_cv['metadatas'][0]}")

# Exemple 2: Recherche vectorielle - CV similaires à une offre
print("\nRecherche de CV similaires à une offre d'emploi")
if job_df.shape[0] > 0:
    # Utiliser le premier job comme exemple
    query_embedding = job_df.iloc[0]['embedding']
    
    results = cv_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    if results and results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]
        
        print(f"  Offre d'exemple: {job_df.iloc[0]['title'][:50]}...")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            similarity = round((1 - dist) * 100, 2)
            print(f"  CV #{i+1} - ID: {meta.get('id', 'N/A')} - "
                  f"Catégorie: {meta.get('Category', 'N/A')} - "
                  f"Similarité: {similarity}%")

# Exemple 3: Recherche vectorielle - Offres similaires à un CV
print("\nRecherche d'offres similaires à un CV")
if resume_df.shape[0] > 0:
    # Utiliser le premier CV comme exemple
    query_embedding = resume_df.iloc[0]['embedding']
    
    results = job_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    if results and results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]
        
        print(f"  CV d'exemple: {resume_df.iloc[0]['cv_id']}")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            similarity = round((1 - dist) * 100, 2)
            print(f"  Offre #{i+1} - ID: {meta.get('id', 'N/A')} - "
                  f"Titre: {meta.get('title', 'N/A')[:50]} - "
                  f"Similarité: {similarity}%")

