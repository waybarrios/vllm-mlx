# Embeddings

vllm-mlx prend en charge les embeddings de texte via [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), en exposant un point d'accès `/v1/embeddings` compatible OpenAI.

## Installation

```bash
pip install mlx-embeddings>=0.0.5
```

## Démarrage rapide

### Lancer le serveur avec un modèle d'embeddings

```bash
# Précharger un modèle d'embeddings spécifique au démarrage
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Si vous n'utilisez pas `--embedding-model`, le modèle d'embeddings est chargé à la demande lors de la première requête, mais uniquement parmi les modèles autorisés par défaut.

### Générer des embeddings avec le SDK OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Texte unique
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# Lot de textes
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=[
        "I love machine learning",
        "Deep learning is fascinating",
        "Natural language processing rocks"
    ]
)
for item in response.data:
    print(f"Text {item.index}: {len(item.embedding)} dimensions")
```

### Utilisation avec curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Modèles pris en charge

Modèles disponibles à la demande :

| Modèle | Cas d'usage | Taille |
|--------|-------------|--------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Rapide et compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | Haute qualité | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Optimal pour l'anglais | Large |
| `mlx-community/multilingual-e5-small-mlx` | Récupération multilingue | Small |
| `mlx-community/multilingual-e5-large-mlx` | Récupération multilingue | Large |
| `mlx-community/bert-base-uncased-mlx` | Référence BERT générale | Base |
| `mlx-community/ModernBERT-base-mlx` | Référence ModernBERT | Base |

Les autres modèles d'embeddings nécessitent l'option `--embedding-model` au démarrage du serveur.

## Gestion des modèles

### Chargement à la demande

Par défaut, le modèle d'embeddings est chargé lors de la première requête sur `/v1/embeddings`. Vous pouvez alterner entre les modèles autorisés listés ci-dessus ; le modèle précédent est déchargé automatiquement.

### Préchargement au démarrage

Utilisez `--embedding-model` pour charger un modèle au démarrage. Lorsque cette option est définie, seul ce modèle peut être utilisé pour les embeddings :

```bash
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Toute requête utilisant un modèle différent renverra une erreur 400.

## Référence de l'API

### POST /v1/embeddings

Génère des embeddings pour le ou les textes fournis.

**Corps de la requête :**

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `model` | string | Oui | Identifiant d'un modèle d'embeddings pris en charge, ou le modèle fixé au démarrage si `--embedding-model` est utilisé |
| `input` | string ou list[string] | Oui | Texte(s) à encoder |

**Réponse :**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]}
  ],
  "model": "mlx-community/all-MiniLM-L6-v2-4bit",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## API Python

### Utilisation directe sans serveur

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## Résolution des problèmes

### mlx-embeddings non installé

```
pip install mlx-embeddings>=0.0.5
```

### Modèle introuvable

Vérifiez que le nom du modèle correspond à l'un des identifiants autorisés listés ci-dessus, ou lancez le serveur avec `--embedding-model` pour fixer un modèle personnalisé. Vous pouvez télécharger les modèles pris en charge à l'avance :

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
