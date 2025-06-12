# Système de matching CV-Offres d'emploi avec RAG

Ce projet implémente un système de correspondance entre CV et offres d'emploi utilisant une approche de Retrieval-Augmented Generation (RAG) basée sur des modèles de langage avancés.

## Architecture du système

Le système suit une architecture RAG en 4 étapes:

1. **Préparation des données**: Extraction et prétraitement des textes de CV et d'offres d'emploi
2. **Génération d'embeddings**: Création de représentations vectorielles des documents
3. **Stockage et récupération vectorielle**: Utilisation d'une base de données vectorielle pour trouver les documents similaires
4. **Génération et évaluation**: Analyse de la correspondance entre CV et offres par un LLM

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
├── 2_embeddings.py            # Script de génération d'embeddings
├── 3_chromadb_retrieval.py    # Script d'indexation dans ChromaDB
├── 4_rag.py                   # Script final pour exécuter le pipeline RAG
├── .env                       # Fichier pour les variables d'environnement (API keys)
├── data/                      # Dossier contenant les données
│   ├── raw/                   # Données brutes & prétraitées (CSV)
│   ├── embeddings/            # Fichiers pickle des embeddings
│   └── chromadb/              # Base de données vectorielle ChromaDB
└── README.md                  # Ce fichier
```

## Fonctionnement du pipeline

### 1. Prétraitement des données (`1_preprocessing.py`)

- Charge les données brutes des CV et offres d'emploi
- Nettoie et structure les textes
- Simplifie le jeu de données en ne gardant que les colonnes essentielles:
  - Pour les CV: `cv_id`, `processed_text`, `Category`
  - Pour les offres: `job_id`, `processed_text`, `title`
- Sauvegarde les données prétraitées

### 2. Génération d'embeddings (`2_embeddings.py`)

- Utilise le modèle `all-MiniLM-L6-v2` de sentence-transformers
- Génère des embeddings de dimension 384 pour chaque document
- Sauvegarde les dataframes avec embeddings dans des fichiers pickle
- Exemple de recherche de similarité pour vérification

### 3. Indexation dans ChromaDB (`3_chromadb_retrieval.py`)

- Crée des collections ChromaDB pour CV et offres d'emploi
- Structure des documents dans ChromaDB:
  - Document: Le texte prétraité
  - Embedding: Le vecteur représentant le document
  - Metadata: ID unique, catégorie (CV) ou titre (offre)
- Batch insertion pour optimiser les performances
- Exemples de requêtes de similarité pour test

### 4. RAG avec LLM (`4_rag.py`)

- Connecte aux collections ChromaDB
- Pour une requête (CV ou offre):
  - Récupère les documents les plus similaires
  - Calcule des scores de similarité normalisés entre 0-100%
  - Prépare un prompt pour le LLM avec les documents récupérés
  - Appelle l'API Hugging Face pour Mistral
  - Affiche les résultats structurés
- Supporte deux modes:
  - CV → Offres d'emploi correspondantes
  - Offre d'emploi → CV correspondants

## Caractéristiques principales

1. **Structure de données optimisée**: Conservation uniquement des colonnes essentielles
2. **ID explicites**: Stockage des identifiants uniques (cv_id, job_id) dans les métadonnées
3. **Scores de similarité intuitifs**: Conversion des distances L2² en pourcentages de similarité
4. **Batch processing**: Insertion par lots pour une meilleure performance
5. **Métadonnées enrichies**: Stockage de métadonnées pertinentes pour faciliter l'interprétation

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
   python 1_preprocessing_v2.py
   ```

2. Générer les embeddings:
   ```bash
   python 2_embeddings_v2.py
   ```

3. Indexer dans ChromaDB:
   ```bash
   python 3_chromadb_retrieval_v2.py
   ```

### Exécution du RAG

Pour lancer le système RAG avec des exemples intégrés:
```bash
python 4_rag.py
```

Pour tester avec vos propres requêtes:
1. Modifier les exemples de CV ou d'offres dans `4_rag.py`
2. Exécuter le script

## Notes techniques

- **Calcul de similarité**: ChromaDB utilise la distance euclidienne au carré (L2²). Nous la convertissons en score de similarité avec la formule: `(1 - distance/2) * 100`
- **Modèle LLM**: Le système utilise Mistral via l'API Hugging Face, mais peut être adapté à d'autres LLMs.
- **Performance**: Le système est optimisé pour équilibrer précision et efficacité. Les requêtes les plus lourdes sont la génération LLM.

## Améliorations possibles

- Intégration de techniques de filtrage préalable par catégories
- Support multilingue pour CV et offres en différentes langues
- Interface utilisateur web pour interroger le système
- Caching des réponses LLM pour les requêtes répétitives
- Fine-tuning du LLM spécifiquement pour l'évaluation CV-offres
