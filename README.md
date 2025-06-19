# Système de matching CV-Offres d'emploi avec RAG

Ce projet implémente un système de correspondance entre CV et offres d'emploi utilisant une approche de Retrieval-Augmented Generation (RAG) basée sur des embeddings sémantiques, ChromaDB et des modèles de langage avancés.

## Architecture du système

Le système suit une architecture RAG en 3 étapes principales :

1. **Prétraitement des données**: Extraction et nettoyage des textes de CV et d'offres d'emploi
2. **Embeddings et stockage vectoriel**: Création de représentations vectorielles des documents et stockage dans ChromaDB avec mesure de similarité cosinus
3. **Recherche contextuelle et génération**: Récupération des documents similaires et analyse par LLM pour déterminer la compatibilité

## Sources de données

Le projet utilise deux ensembles de données de Kaggle :

- **CV/Resume Dataset** : https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data
  - Contient des CV avec des informations telles que catégories, compétences et descriptions

- **LinkedIn Job Postings Dataset** : https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
  - Contient des offres d'emploi avec titres, descriptions et exigences

## Structure du projet

```
Azure_ModelLLM/
├── 1_preprocessing.py         # Script de prétraitement des données
├── 2_embeddings_chroma.py     # Script combiné de génération d'embeddings et stockage ChromaDB
├── 3_rag.py                   # Script RAG pour le matching CV-Offres
├── .env                       # Fichier pour les variables d'environnement (API keys)
├── data/                      # Dossier contenant les données
│   ├── raw/                   # Données brutes & prétraitées (CSV)
│   └── chromadb/              # Base de données vectorielle ChromaDB (stockage persistant)
└── README.md                  # Ce fichier
```

## Fonctionnement du pipeline

### 1. Prétraitement des données (`1_preprocessing.py`)

- Charge les données brutes des CV et offres d'emploi
- Nettoie et structure les textes 
- Simplifie le jeu de données en ne gardant que les colonnes essentielles:
  - Pour les CV: `cv_id`, `processed_text`, `Category`
  - Pour les offres: `job_id`, `processed_text`, `title`
- Sauvegarde les données prétraitées en CSV

### 2. Embeddings et stockage ChromaDB (`2_embeddings_chroma.py`)

- Charge les données prétraitées
- Utilise `SentenceTransformerEmbeddingFunction` avec le modèle `all-MiniLM-L6-v2`
- Crée des collections ChromaDB avec **métrique de similarité cosinus** explicite
- Structure des documents dans ChromaDB:
  - Document: Le texte prétraité
  - Embedding: Généré automatiquement par la fonction d'embedding
  - Metadata: Catégorie (CV) ou titre (offre)
  - ID: Identifiant unique des CV/offres
- Insertion par batch pour optimiser les performances
- Inclut des fonctions de diagnostic pour vérifier les performances du matching
- Teste des exemples de requêtes par texte et par embedding

### 3. RAG avec LLM (`3_rag.py`)

- Connecte aux collections ChromaDB existantes
- Définit une fonction optimisée `retrieve_top_k` qui:
  - Gère les requêtes par texte ou embedding
  - Convertit la distance cosinus en score de similarité: `(1 - distance/2) * 100`
  - Traite correctement les métadonnées et IDs retournés par ChromaDB
- Propose deux fonctions principales:
  - `ask_for_best_jobs`: CV → Offres d'emploi correspondantes
  - `ask_for_best_cvs`: Offre d'emploi → CV correspondants
- Génère des réponses en français via l'API Hugging Face pour Mixtral

## Caractéristiques principales

1. **Métrique cosinus explicite**: Utilisation explicite de la similarité cosinus dans ChromaDB via `metadata={"hnsw:space": "cosine"}`
2. **Gestion robuste des métadonnées**: Traitement correct de la structure particulière des métadonnées retournées par ChromaDB
3. **IDs correctement associés**: Récupération et affichage des identifiants uniques des documents dans les résultats
4. **Conversion de distance optimisée**: Transformation de la distance cosinus en score de similarité intuitif
5. **Batch processing**: Insertion par lots pour une meilleure performance
6. **Pipeline RAG intégré**: Connexion transparente entre la recherche vectorielle et la génération contextuelle

## Configuration requise

### Prérequis

- Python 3.8+
- Accès à l'API Hugging Face (token)
- Librairies requises (voir requirements.txt)

### Variables d'environnement

Créer un fichier `.env` à la racine du projet avec:

```
HF_API_TOKEN=votre_token_hugging_face
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Préparation du pipeline complet

1. Exécuter le prétraitement:
   ```bash
   python 1_preprocessing.py
   ```

2. Générer les embeddings et créer les collections ChromaDB:
   ```bash
   python 2_embeddings_chroma.py
   ```

### Exécution du RAG

Pour lancer le système RAG avec des exemples intégrés:
```bash
python 3_rag.py
```

Pour tester avec vos propres requêtes:
1. Modifier les exemples de CV ou d'offres dans `3_rag.py`
2. Exécuter le script

## Notes techniques

- **Métrique de similarité**: ChromaDB est configuré pour utiliser la **similarité cosinus** au lieu de la distance euclidienne par défaut
- **Calcul de similarité**: Conversion de la distance cosinus (0 à 2) en pourcentage de similarité avec la formule: `(1 - distance/2) * 100`
- **Structure des métadonnées**: ChromaDB retourne les métadonnées sous forme d'une liste de dictionnaires individuels, nécessitant un traitement adapté
- **Modèle LLM**: Le système utilise Mixtral via l'API Hugging Face, configuré pour générer des réponses en français
- **Performance**: Les scores de similarité avec la métrique cosinus atteignent généralement >70% pour les correspondances pertinentes

## Améliorations possibles

- Intégration de techniques de filtrage préalable par catégories (pré-filtrage)
- Adaptation du contexte fourni au LLM en fonction du type de requête
- Support multilingue pour CV et offres en différentes langues
- Interface utilisateur web pour interroger le système
- Caching des réponses LLM pour les requêtes répétitives
- Possibilité de feedback utilisateur pour améliorer les correspondances
- Ajout d'un système d'explication de scores de similarité plus détaillé
