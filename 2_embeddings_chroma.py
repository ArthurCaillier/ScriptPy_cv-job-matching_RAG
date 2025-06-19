
"""
Script combiné qui génère des embeddings à partir des données prétraitées
et les stocke directement dans ChromaDB.
"""
# Chargement packages
import pandas as pd
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

#*------ Chargement des fichiers prétraités ------*#
job_df = pd.read_csv("data/processed_jobs.csv")
resume_df = pd.read_csv("data/processed_resumes.csv")
print(f"Données prétraitées chargées: {len(job_df)} offres et {len(resume_df)} CV")

#*------ Initialisation ChromaDB ------*#
# Configuration
CHROMA_DIR = "data/chromadb"
os.makedirs(CHROMA_DIR, exist_ok=True)
# Initialisation 
client = chromadb.PersistentClient(path=CHROMA_DIR)

## Suppression des collections existantes si elles existent
# try:
#     client.delete_collection("cv_collection")
#     client.delete_collection("job_collection")
#     print("Collections existantes supprimées")
# except:
#     print("Pas de collections existantes à supprimer")

# Fonction d'embedding (sentence-transformers)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

#*------ Création des collections ------*#
cv_collection = client.create_collection(
    name="cv_collection",
    embedding_function=embedding_function,
    metadata={"description": "Collection de CV embeddings",
              "hnsw:space": "cosine"  # Spécification explicite de la métrique cosinus
                                      # Tous les documents ajoutés seront indexés selon cette métrique
              }
)

job_collection = client.create_collection(
    name="job_collection",
    embedding_function=embedding_function,
    metadata={"description": "Collection d'offres d'emploi embeddings",
              "hnsw:space": "cosine"}
)

print(f"Métrique CV collection: {cv_collection.metadata.get('hnsw:space', 'non spécifiée')}")
print(f"Métrique Job collection: {job_collection.metadata.get('hnsw:space', 'non spécifiée')}")

# #!Notes! Après création des collections, il est possible d'y accéder directement via :
# cv_collection = client.get_collection(name="cv_collection", embedding_function=embedding_function)
# job_collection = client.get_collection(name="job_collection", embedding_function=embedding_function)
# # Il est essentiel de recharger la fonction d'embedding pour réaliser les requêtes textuelles (fin de script)

#*------ Remplissage des collections ------*#
# Fonction pour ajouter des documents à une collection avec gestion des batches
def add_documents_to_collection(collection, df, id_col, text_col, 
                                metadata_cols=None, batch_size=100):
    """
    Ajoute des documents à une collection ChromaDB par batches.
    
    Args:
        collection: Collection ChromaDB
        df: DataFrame contenant les documents
        id_col: Nom de la colonne contenant les identifiants
        text_col: Nom de la colonne contenant le texte à encoder
        metadata_cols: Liste des colonnes à inclure dans les métadonnées
        batch_size: Taille des batches pour l'insertion
    """
    total_documents = len(df)
    print(f"Ajout de {total_documents} documents à la collection {collection.name}...")
    
    if metadata_cols is None:
        metadata_cols = []
    
    for i in tqdm(range(0, total_documents, batch_size)):
        end_idx = min(i + batch_size, total_documents)
        batch_df = df.iloc[i:end_idx]
        
        # Construction des listes pour l'ajout en batch
        ids = batch_df[id_col].astype(str).tolist()
        texts = batch_df[text_col].tolist()
        
        # Construction des métadonnées si demandé
        metadatas = None
        if metadata_cols:
            metadatas = []
            for _, row in batch_df.iterrows():
                metadata = {col: str(row[col]) for col in metadata_cols if col in row}
                metadatas.append(metadata)
        
        # Ajout du batch à la collection (les embeddings sont générés automatiquement)
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"Tous les documents ont été ajoutés à la collection {collection.name}")

# Ajout des CV à la collection
add_documents_to_collection(
    cv_collection,
    resume_df,
    id_col="cv_id",
    text_col="processed_text",
    metadata_cols=["Category"]
)

# Ajout des offres d'emploi à la collection
add_documents_to_collection(
    job_collection,
    job_df,
    id_col="job_id",
    text_col="processed_text",
    metadata_cols=["title"]
)

#*------ Statistiques des collections ------*#
print(f"Collection CV: {cv_collection.count()} documents")
print(f"Collection Offres: {job_collection.count()} documents")

# Exemple de CV
print("\n• Exemple d'entrée de CV:")
example_cv = cv_collection.get(limit=1)
cv_id = example_cv['ids'][0]
cv_metadata = example_cv['metadatas'][0]
print(f"  - ID: {cv_id}")
print(f"  - Métadonnées: {cv_metadata}")
print(f"  - Extrait du document: {example_cv['documents'][0][:100]}...")

# Exemple d'offre
print("\n• Exemple d'entrée d'offre d'emploi:")
example_job = job_collection.get(limit=1)
job_id = example_job['ids'][0]
job_metadata = example_job['metadatas'][0]
print(f"  - ID: {job_id}")
print(f"  - Métadonnées: {job_metadata}")
print(f"  - Extrait du document: {example_job['documents'][0][:100]}...")

#*------ Test de requête de texte à embedding ------*#
print("\nTest de requête sur la collection des CV...")
cv_results = cv_collection.query(
    query_texts=["python developper data science machine learning"],
    n_results=3
)
print(f"Résultats pour la requête 'python developper data science machine learning':")
for i, (doc_id, doc, distance, metadata) in enumerate(zip(
    cv_results['ids'][0],
    cv_results['documents'][0],
    cv_results['distances'][0],
    cv_results['metadatas'][0]
)):
    # Calcul de la similarité à partir de la distance cosinus
    similarity = (1 - distance/2) * 100  # Formule linéaire pour la conversion
    print(f"CV {i+1}: ID={doc_id}, Similarité={similarity:.2f}%")
    print(f"Catégorie: {metadata.get('Category', 'N/A')}")
    print(f"Extrait du texte: {doc[:100]}...\n")

print("\nTest de requête sur la collection des offres d'emploi...")
job_results = job_collection.query(
    query_texts=["development engineer web fullstack"],
    n_results=3
)
print(f"Résultats pour la requête 'development engineer web fullstack':")
for i, (doc_id, doc, distance, metadata) in enumerate(zip(
    job_results['ids'][0],
    job_results['documents'][0],
    job_results['distances'][0],
    job_results['metadatas'][0]
)):
    # Calcul de la similarité à partir de la distance cosinus
    similarity = (1 - distance/2) * 100  # Formule linéaire pour la conversion
    print(f"Offre {i+1}: ID={doc_id}, Similarité={similarity:.2f}%")
    print(f"Titre: {metadata.get('title', 'N/A')}")
    print(f"Extrait du texte: {doc[:100]}...\n")


#*------ Test de matching CV-Offres (embedding à embedding) ------*#
# Exemple 1: Recherche de CV similaires à une offre spécifique
# Récupérer la première offre comme exemple
example_job = job_collection.get(limit=1)
job_id = example_job['ids'][0]
job_embedding = embedding_function(example_job['documents'])[0]
job_title = example_job['metadatas'][0].get('title', 'N/A')

# Rechercher les CV similaires (ici 3) à cette offre
results = cv_collection.query(
    query_embeddings=[job_embedding],
    n_results=3
)
print(f"• Offre d'exemple: ID={job_id}, Titre={job_title}")
print("• CV similaires à cette offre:")
for i, (cv_id, distance, metadata) in enumerate(zip(
    results['ids'][0],
    results['distances'][0],
    results['metadatas'][0]
)):
    # Appliquer notre formule linéaire pour la similarité
    similarity = (1 - distance/2) * 100
    print(f"  - CV #{i+1}: ID={cv_id}, Similarité={similarity:.2f}%")
    print(f"    Catégorie: {metadata.get('Category', 'N/A')}")

# Exemple 2: Recherche d'offres similaires à un CV spécifique
# Récupérer le premier CV comme exemple
example_cv = cv_collection.get(limit=1)
cv_id = example_cv['ids'][0]
cv_embedding = embedding_function(example_cv['documents'][0])
cv_category = example_cv['metadatas'][0].get('Category', 'N/A')

# Rechercher les offres similaires à ce CV
results = job_collection.query(
    query_embeddings=[cv_embedding],
    n_results=3
)
print(f"• CV d'exemple: ID={cv_id}, Catégorie={cv_category}")
print("• Offres similaires à ce CV:")
for i, (job_id, distance, metadata_dict) in enumerate(zip(
    results['ids'][0],
    results['distances'][0],
    results['metadatas'][0] 
)):
    # Appliquer notre formule linéaire pour la similarité
    similarity = (1 - distance/2) * 100
    print(f"  - Offre #{i+1}: ID={job_id}, Similarité={similarity:.2f}%")
    print(f"    Titre: {metadata_dict.get('title', 'N/A')}")


