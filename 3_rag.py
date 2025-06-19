# Chargement package
import os
import requests
import chromadb
import numpy as np
from dotenv import load_dotenv
from chromadb.utils import embedding_functions


#*------ Configuration Hugging Face API ------*#
# Chargement de la clé stocké depuis .env
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Initialisation de la fonction d'embedding avec le même modèle que dans 2_embeddings_chroma.py
# IMPORTANT: Il faut utiliser exactement la même fonction d'embedding pour la cohérence
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

print("Connexion aux collections ChromaDB...")
try:
    client = chromadb.PersistentClient(path="data/chromadb")
    # IMPORTANT: Toujours passer la fonction d'embedding lors de la récupération des collections
    cv_collection = client.get_collection("cv_collection", embedding_function=embedding_function)
    job_collection = client.get_collection("job_collection", embedding_function=embedding_function)
    print(f"→ Collections chargées: {cv_collection.count()} CV, {job_collection.count()} offres")
except Exception as e:
    print(f"❌ Erreur lors du chargement des collections: {str(e)}")
    print("Vérifiez que le dossier data/chromadb existe et contient les collections.")

#*------ Fonction de récupération des documents les plus proches ------*#
def retrieve_top_k(collection, query_text=None, query_embedding=None, k=5):
    """
    Récupère les k documents les plus proches dans la collection.
    
    Args:
        collection: Collection ChromaDB
        query_text: Texte de la requête (si fourni, sera converti en embedding)
        query_embedding: Embedding déjà calculé (si query_text n'est pas fourni)
        k: Nombre de résultats à récupérer
        
    Returns:
        Liste de documents avec leurs métadonnées enrichies
    """
    if query_text is not None:
        # Si on passe du texte, ChromaDB utilise automatiquement embedding_function
        results = collection.query(
            query_texts=[query_text], 
            n_results=k,
            include=["documents", "metadatas", "distances"]  # "ids" est toujours inclus par défaut
        )
    elif query_embedding is not None:
        # Si on passe un embedding, s'assurer qu'il est au format liste
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=k,
            include=["documents", "metadatas", "distances"]  # "ids" est toujours inclus par défaut
        )
    else:
        raise ValueError("Vous devez fournir soit query_text soit query_embedding")
    
    documents = []
    for i, (doc, meta, dist, doc_id) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0],
        results["ids"][0]  
    )):
        # Conversion de la distance cosinus en similarité
        similarity = round((1 - dist / 2) * 100, 2)
        
        meta_with_scores = meta.copy()
        meta_with_scores["distance"] = dist
        meta_with_scores["similarity"] = similarity
        documents.append({
            "text": doc, 
            "metadata": meta_with_scores,
            "id": doc_id  
        })
    
    return documents

#*------ Fonction d'appel à l'API Mistral via Hugging Face ------*#
def generate_response(context, query):
    prompt = f"""<s>[INST] Tu es un assistant de recrutement spécialisé 
    dans le matching entre CV et offres d'emploi. Réponds en français.
    Voici des documents pertinents :

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

#*------ Exemple 1 : Trouver les meilleurs CV pour un poste ------*#
def ask_for_best_cvs(job_description):
    """Trouve les CV les plus adaptés pour une description de poste."""
    top_cvs = retrieve_top_k(cv_collection, query_text=job_description, k=5)
    
    print(f"\n💼 {len(top_cvs)} CV similaires trouvés dans ChromaDB:")
    for i, cv in enumerate(top_cvs):
        metadata = cv["metadata"]
        cv_id = cv["id"]  
        category = metadata.get("Category", "N/A")
        similarity = metadata.get("similarity", "N/A")
        print(f"CV #{i+1} - ID: {cv_id} - Catégorie: {category} - Similarité: {similarity}% - Début: {cv['text'][:100]}...")
    
    # Préparer le contexte et générer la réponse
    context = "\n\n".join([cv["text"] for cv in top_cvs])
    return generate_response(context, f"Voici une description de poste: {job_description}\n\nIdentifie parmi les CV disponibles celui ou ceux qui correspondent le mieux à ce poste. Explique pourquoi ils sont pertinents en détaillant les compétences clés qui correspondent et les éventuelles lacunes.")

#*------ Exemple 2 : Trouver les meilleures offres pour un CV ------*#
def ask_for_best_jobs(cv_text):
    """Trouve les offres d'emploi les plus adaptées pour un CV."""
    top_jobs = retrieve_top_k(job_collection, query_text=cv_text, k=5)
    
    print(f"\n💼 {len(top_jobs)} offres d'emploi similaires trouvées dans ChromaDB:")
    for i, job in enumerate(top_jobs):
        metadata = job["metadata"]
        job_id = job["id"]
        title = metadata.get("title", "Sans titre")
        similarity = metadata.get("similarity", "N/A")
        print(f"Offre #{i+1} - ID: {job_id} - Titre: {title} - Similarité: {similarity}% - Début: {job['text'][:100]}...")
    
    # Préparer le contexte et générer la réponse
    context = "\n\n".join([job["text"] for job in top_jobs])
    return generate_response(context, f"Voici un CV: {cv_text}\n\nIdentifie parmi les offres d'emploi disponibles celle(s) qui correspondent le mieux à ce profil. Détaille pour chaque offre pertinente les points forts du candidat et les éventuelles compétences à développer.")

#*------ Exemples de test ------*#
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
