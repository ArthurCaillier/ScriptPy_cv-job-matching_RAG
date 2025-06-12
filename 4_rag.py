import os
import requests
import chromadb
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

#*------ Chargement des variables d'environnement depuis .env ------*#
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

#*------ Configuration Hugging Face API ------*#
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Initialisation du modèle d'embeddings et client ChromaDB
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Connexion aux collections ChromaDB...")
try:
    client = chromadb.PersistentClient(path="data/chromadb")
    cv_collection = client.get_collection("cv_collection")
    job_collection = client.get_collection("job_collection")
    print(f"→ Collections chargées: {cv_collection.count()} CV, {job_collection.count()} offres")
except Exception as e:
    print(f"❌ Erreur lors du chargement des collections: {str(e)}")
    print("Vérifiez que le dossier data/chromadb existe et contient les collections.")

#*------ Fonction de récupération des documents les plus proches ------*#
def retrieve_top_k(collection, query_embedding, k=5):
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    )):
        # Conversion de la distance L2 au carré en similarité avec une formule linéaire
        # Comme la distance L2² max entre deux vecteurs unitaires est 2, cette formule donne:
        # - 100% pour des documents identiques (dist = 0)
        # - 0% pour des documents orthogonaux (dist = 2)
        similarity = round((1 - dist / 2) * 100, 2)
        
        meta["distance"] = dist  # Ajouter la distance aux métadonnées
        meta["similarity"] = similarity  # Ajouter le score de similarité en pourcentage
        documents.append({"text": doc, "metadata": meta})
    
    return documents

# --- Fonction d'appel à l'API Mistral via Hugging Face ---
def generate_response(context, query):
    prompt = f"""<s>[INST] Tu es un assistant de recrutement spécialisé dans le matching entre CV et offres d'emploi. Voici des documents pertinents :

{context}

Sur base de ces documents : {query} [/INST]</s>"""

    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"].split("[/INST]</s>")[1].strip()
        else:
            error_info = response.json() if response.content else {"error": "Unknown error"}
            if response.status_code == 503 and "estimated_time" in error_info:
                wait_time = error_info["estimated_time"]
                return f"Modèle en cours de chargement, temps estimé: {wait_time} secondes. Veuillez réessayer plus tard."
            return f"❌ Erreur API ({response.status_code}): {error_info}"
            
    except Exception as e:
        return f"❌ Erreur lors de la génération: {str(e)}"

# --- Exemple 1 : Trouver les meilleurs CV pour un poste ---
def ask_for_best_cvs(job_description):
    """Trouve les CV les plus adaptés pour une description de poste."""
    # Générer l'embedding pour la requête
    query_embedding = model.encode(job_description)
    
    # Récupérer les CV les plus similaires
    top_cvs = retrieve_top_k(cv_collection, query_embedding, k=5)
    
    print(f"\n💼 {len(top_cvs)} CV similaires trouvés dans ChromaDB:")
    for i, cv in enumerate(top_cvs):
        metadata = cv["metadata"]
        id_info = metadata.get("id", "N/A")
        category = metadata.get("Category", "N/A")
        similarity = metadata.get("similarity", "N/A")
        print(f"CV #{i+1} - ID: {id_info} - Catégorie: {category} - Similarité: {similarity}% - Début: {cv['text'][:100]}...")
    
    # Préparer le contexte et générer la réponse
    context = "\n\n".join([cv["text"] for cv in top_cvs])
    return generate_response(context, f"Voici une description de poste: {job_description}\n\nIdentifie parmi les CV disponibles celui ou ceux qui correspondent le mieux à ce poste. Explique pourquoi ils sont pertinents en détaillant les compétences clés qui correspondent et les éventuelles lacunes.")

# --- Exemple 2 : Trouver les meilleures offres pour un CV ---
def ask_for_best_jobs(cv_text):
    """Trouve les offres d'emploi les plus adaptées pour un CV."""
    # Générer l'embedding pour la requête
    query_embedding = model.encode(cv_text)
    
    # Récupérer les offres les plus similaires
    top_jobs = retrieve_top_k(job_collection, query_embedding, k=5)
    
    print(f"\n💼 {len(top_jobs)} offres d'emploi similaires trouvées dans ChromaDB:")
    for i, job in enumerate(top_jobs):
        metadata = job["metadata"]
        id_info = metadata.get("id", "N/A")
        title = metadata.get("title", "Sans titre")
        similarity = metadata.get("similarity", "N/A")
        print(f"Offre #{i+1} - ID: {id_info} - Titre: {title} - Similarité: {similarity}% - Début: {job['text'][:100]}...")
    
    # Préparer le contexte et générer la réponse
    context = "\n\n".join([job["text"] for job in top_jobs])
    return generate_response(context, f"Voici un CV: {cv_text}\n\nIdentifie parmi les offres d'emploi disponibles celle(s) qui correspondent le mieux à ce profil. Détaille pour chaque offre pertinente les points forts du candidat et les éventuelles compétences à développer.")

# --- Exemples de test ---
if __name__ == "__main__":
    # Test 1 : poste donné → CV
    job_description = """
    Poste: Data Scientist avec expérience en NLP
    Description: Nous recherchons un Data Scientist expérimenté dans le traitement du langage naturel (NLP) pour rejoindre notre équipe de développement d'IA.
    Compétences requises:
    - Python avancé
    - Expérience avec des modèles de language comme BERT, GPT
    - Bonnes compétences en MLOps
    - Connaissance des techniques de vectorisation et embeddings
    """
    print("\nOffre > CV")
    response = ask_for_best_cvs(job_description)
    print(response)
    
    # Test 2 : CV donné → offres
    cv_text = """
    Ingénieur en développement logiciel avec 5 ans d'expérience.
    Compétences:
    - Python, Java, C++
    - Développement web (HTML, CSS, JavaScript, React)
    - DevOps (Docker, Kubernetes)
    - Cloud (AWS, Azure)
    Expériences:
    - Développement d'applications web responsives
    - Mise en place de pipelines CI/CD
    - Optimisation de performance d'applications
    """
    print("\n CV > Offres")
    response = ask_for_best_jobs(cv_text)
    print(response)