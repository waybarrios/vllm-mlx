# Embeddings

vllm-mlx soporta embeddings de texto usando [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), y expone un endpoint `/v1/embeddings` compatible con OpenAI.

## Instalacion

```bash
pip install mlx-embeddings>=0.0.5
```

## Inicio rápido

### Iniciar el servidor con un modelo de embeddings

```bash
# Precarga un modelo de embeddings especifico al inicio
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Si no se usa `--embedding-model`, el modelo de embeddings se carga de forma diferida en la primera solicitud, pero solo desde la lista de modelos permitidos en tiempo de solicitud.

### Generar embeddings con el SDK de OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Texto individual
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# Lote de textos
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

### Usando curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Modelos soportados

Modelos disponibles en tiempo de solicitud:

| Model | Use Case | Size |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Fast, compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | High quality | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Best for English | Large |
| `mlx-community/multilingual-e5-small-mlx` | Multilingual retrieval | Small |
| `mlx-community/multilingual-e5-large-mlx` | Multilingual retrieval | Large |
| `mlx-community/bert-base-uncased-mlx` | General BERT baseline | Base |
| `mlx-community/ModernBERT-base-mlx` | ModernBERT baseline | Base |

Otros modelos de embeddings requieren `--embedding-model` al iniciar el servidor.

## Gestion de modelos

### Carga diferida

Por defecto, el modelo de embeddings se carga en la primera solicitud a `/v1/embeddings`. Es posible cambiar entre los modelos permitidos en tiempo de solicitud, y el modelo anterior se descarga automaticamente.

### Precarga al inicio

Usa `--embedding-model` para cargar un modelo al iniciar el servidor. Cuando se establece esta opcion, solo ese modelo puede usarse para embeddings:

```bash
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Solicitar un modelo diferente devolvera un error 400.

## Referencia de la API

### POST /v1/embeddings

Crea embeddings para los textos de entrada proporcionados.

**Cuerpo de la solicitud:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Supported embedding model ID, or the startup-pinned model when `--embedding-model` is used |
| `input` | string or list[string] | Yes | Text(s) to embed |

**Respuesta:**

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

## API de Python

### Uso directo sin servidor

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## Solucion de problemas

### mlx-embeddings no esta instalado

```
pip install mlx-embeddings>=0.0.5
```

### Modelo no encontrado

Asegurate de que el nombre del modelo coincida con alguno de los IDs permitidos en tiempo de solicitud, o inicia el servidor con `--embedding-model` para fijar un modelo personalizado. Puedes descargar los modelos soportados con anticipacion:

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
