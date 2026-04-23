# Serveur compatible OpenAI

vllm-mlx fournit un serveur FastAPI avec une compatibilité complète avec l'API OpenAI.

Par défaut, le serveur n'écoute que sur `127.0.0.1`. Utilisez `--host 0.0.0.0` uniquement si vous souhaitez délibérément l'exposer au-delà de la machine locale.

## Démarrage du serveur

### Mode simple (par défaut)

Débit maximal pour un utilisateur unique :

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
```

### Mode continuous batching

Pour plusieurs utilisateurs simultanés :

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

### Avec paged cache

Mise en cache efficace en mémoire pour la production :

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching --use-paged-cache
```

## Options du serveur

| Option | Description | Défaut |
|--------|-------------|--------|
| `--port` | Port du serveur | 8000 |
| `--host` | Hôte du serveur | 127.0.0.1 |
| `--api-key` | Clé API pour l'authentification | None |
| `--rate-limit` | Requêtes par minute par client (0 = désactivé) | 0 |
| `--timeout` | Délai d'expiration des requêtes en secondes | 300 |
| `--enable-metrics` | Expose les métriques Prometheus sur `/metrics` | False |
| `--continuous-batching` | Active le batching pour plusieurs utilisateurs | False |
| `--use-paged-cache` | Active le paged KV cache | False |
| `--cache-memory-mb` | Limite mémoire du cache en Mo | Auto |
| `--cache-memory-percent` | Fraction de la RAM réservée au cache | 0.20 |
| `--max-tokens` | Nombre maximal de tokens par défaut | 32768 |
| `--max-request-tokens` | Valeur maximale de `max_tokens` acceptée des clients API | 32768 |
| `--default-temperature` | Température par défaut si non spécifiée | None |
| `--default-top-p` | Valeur top_p par défaut si non spécifiée | None |
| `--stream-interval` | Tokens par fragment de streaming | 1 |
| `--mcp-config` | Chemin vers le fichier de configuration MCP | None |
| `--reasoning-parser` | Parser pour les modèles reasoning (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Précharge un modèle d'embeddings au démarrage | None |
| `--enable-auto-tool-choice` | Active le tool calling automatique | False |
| `--tool-call-parser` | Parser de tool calling (voir [Tool Calling](tool-calling.md)) | None |

## Points de terminaison de l'API

### Chat Completions

```bash
POST /v1/chat/completions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Completions

```bash
POST /v1/completions
```

```python
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    max_tokens=50
)
```

### Modèles

```bash
GET /v1/models
```

Retourne les modèles disponibles.

### Embeddings

```bash
POST /v1/embeddings
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

Voir le [Guide des embeddings](embeddings.md) pour plus de détails.

### Vérification de l'état

```bash
GET /health
```

Retourne l'état du serveur.

### Métriques

```bash
GET /metrics
```

Point de terminaison de collecte Prometheus pour les métriques du serveur, du cache, du scheduler et des requêtes.
Le point de terminaison est désactivé par défaut et s'active avec `--enable-metrics`.

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-metrics
```

`/metrics` est intentionnellement non authentifié. Exposez-le uniquement sur un réseau de confiance ou derrière un reverse proxy ou un pare-feu qui limite les accès.

### API Anthropic Messages

```bash
POST /v1/messages
```

Point de terminaison compatible Anthropic qui permet à des outils comme Claude Code et OpenCode de se connecter directement à vllm-mlx. En interne, il traduit les requêtes Anthropic au format OpenAI, exécute l'inférence via le moteur, puis convertit la réponse au format Anthropic.

Fonctionnalités :
- Réponses non-streaming et streaming (SSE)
- Messages système (chaîne simple ou liste de blocs de contenu)
- Conversations multi-tours avec messages utilisateur et assistant
- Tool calling avec blocs de contenu `tool_use` et `tool_result`
- Comptage de tokens pour le suivi du budget
- Contenu multimodal (images via blocs `source`)
- Détection de déconnexion client (retourne HTTP 499)
- Filtrage automatique des tokens spéciaux dans la sortie en streaming

#### Non-streaming

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
# Response includes: response.id, response.model, response.stop_reason,
# response.usage.input_tokens, response.usage.output_tokens
```

#### Streaming

Le streaming suit le protocole d'événements SSE d'Anthropic. Les événements sont émis dans cet ordre :
`message_start` -> `content_block_start` -> `content_block_delta` (répété) -> `content_block_stop` -> `message_delta` -> `message_stop`

```python
with client.messages.stream(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

#### Messages système

Les messages système peuvent être une chaîne simple ou une liste de blocs de contenu :

```python
# Plain string
response = client.messages.create(
    model="default",
    max_tokens=256,
    system="You are a helpful coding assistant.",
    messages=[{"role": "user", "content": "Write a hello world in Python"}]
)

# List of content blocks
response = client.messages.create(
    model="default",
    max_tokens=256,
    system=[
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Be concise in your answers."},
    ],
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

#### Tool calling

Définissez les outils avec `name`, `description` et `input_schema`. Le modèle retourne des blocs de contenu `tool_use` lorsqu'il souhaite appeler un outil. Renvoyez les résultats sous forme de blocs `tool_result`.

```python
# Step 1: Send request with tools
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)

# Step 2: Check if model wants to use tools
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}, ID: {block.id}")
        # response.stop_reason will be "tool_use"

# Step 3: Send tool result back
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Sunny, 22C"
            }
        ]}
    ],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)
print(response.content[0].text)  # "The weather in Paris is sunny, 22C."
```

Modes de sélection d'outil :

| `tool_choice` | Comportement |
|---------------|--------------|
| `{"type": "auto"}` | Le modèle décide d'appeler ou non des outils (par défaut) |
| `{"type": "any"}` | Le modèle doit appeler au moins un outil |
| `{"type": "tool", "name": "get_weather"}` | Le modèle doit appeler l'outil spécifié |
| `{"type": "none"}` | Le modèle n'appellera aucun outil |

#### Conversations multi-tours

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

response = client.messages.create(
    model="default",
    max_tokens=100,
    messages=messages
)
```

#### Comptage de tokens

```bash
POST /v1/messages/count_tokens
```

Compte les tokens d'entrée d'une requête Anthropic en utilisant le tokenizer du modèle. Utile pour le suivi du budget avant d'envoyer une requête. Comptabilise les tokens des messages système, des messages de conversation, des entrées `tool_use`, du contenu `tool_result` et des définitions d'outils (name, description, input_schema).

```python
import requests

resp = requests.post("http://localhost:8000/v1/messages/count_tokens", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "system": "You are helpful.",
    "tools": [{
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    }]
})
print(resp.json())  # {"input_tokens": 42}
```

#### Exemples curl

Non-streaming :

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming :

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a joke"}]
  }'
```

Comptage de tokens :

```bash
curl http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
# {"input_tokens": 12}
```

#### Champs de la requête

| Champ | Type | Requis | Défaut | Description |
|-------|------|--------|--------|-------------|
| `model` | string | oui | - | Nom du modèle (utilisez `"default"` pour le modèle chargé) |
| `messages` | list | oui | - | Messages de conversation avec `role` et `content` |
| `max_tokens` | int | oui | - | Nombre maximal de tokens à générer |
| `system` | string or list | non | null | Invite système (chaîne ou liste de blocs `{"type": "text", "text": "..."}`) |
| `stream` | bool | non | false | Active le streaming SSE |
| `temperature` | float | non | 0.7 | Température d'échantillonnage (0.0 = déterministe, 1.0 = créatif) |
| `top_p` | float | non | 0.9 | Seuil d'échantillonnage nucleus |
| `top_k` | int | non | null | Échantillonnage top-k |
| `stop_sequences` | list | non | null | Séquences qui arrêtent la génération |
| `tools` | list | non | null | Définitions d'outils avec `name`, `description`, `input_schema` |
| `tool_choice` | dict | non | null | Mode de sélection d'outil (`auto`, `any`, `tool`, `none`) |
| `metadata` | dict | non | null | Métadonnées arbitraires (transmises telles quelles, non utilisées par le serveur) |

#### Format de réponse

Réponse non-streaming :

```json
{
  "id": "msg_abc123...",
  "type": "message",
  "role": "assistant",
  "model": "default",
  "content": [
    {"type": "text", "text": "Hello! How can I help?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  }
}
```

Lorsque des outils sont appelés, `content` inclut des blocs `tool_use` et `stop_reason` vaut `"tool_use"` :

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "get_weather",
      "input": {"city": "Paris"}
    }
  ],
  "stop_reason": "tool_use"
}
```

Raisons d'arrêt :

| `stop_reason` | Signification |
|---------------|---------------|
| `end_turn` | Le modèle a terminé naturellement |
| `tool_use` | Le modèle souhaite appeler un outil |
| `max_tokens` | La limite `max_tokens` a été atteinte |

#### Utilisation avec Claude Code

Pointez Claude Code directement vers votre serveur vllm-mlx :

```bash
# Start the server
vllm-mlx serve mlx-community/Qwen3-Coder-Next-235B-A22B-4bit \
  --continuous-batching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# In another terminal, configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### État du serveur

```bash
GET /v1/status
```

Point de terminaison de surveillance en temps réel qui retourne des statistiques globales du serveur et des détails par requête. Utile pour déboguer les performances, suivre l'efficacité du cache et surveiller la mémoire Metal GPU.

```bash
curl -s http://localhost:8000/v1/status | python -m json.tool
```

Exemple de réponse :

```json
{
  "status": "running",
  "model": "mlx-community/Qwen3-8B-4bit",
  "uptime_s": 342.5,
  "steps_executed": 1247,
  "num_running": 1,
  "num_waiting": 0,
  "total_requests_processed": 15,
  "total_prompt_tokens": 28450,
  "total_completion_tokens": 3200,
  "metal": {
    "active_memory_gb": 5.2,
    "peak_memory_gb": 8.1,
    "cache_memory_gb": 2.3
  },
  "cache": {
    "type": "memory_aware_cache",
    "entries": 5,
    "hit_rate": 0.87,
    "memory_mb": 2350
  },
  "requests": [
    {
      "request_id": "req_abc123",
      "phase": "generation",
      "tokens_per_second": 45.2,
      "ttft_s": 0.8,
      "progress": 0.35,
      "cache_hit_type": "prefix",
      "cached_tokens": 1200,
      "generated_tokens": 85,
      "max_tokens": 256
    }
  ]
}
```

Champs de la réponse :

| Champ | Description |
|-------|-------------|
| `status` | État du serveur : `running`, `stopped` ou `not_loaded` |
| `model` | Nom du modèle chargé |
| `uptime_s` | Secondes écoulées depuis le démarrage du serveur |
| `steps_executed` | Nombre total d'étapes d'inférence exécutées |
| `num_running` | Nombre de requêtes en cours de génération de tokens |
| `num_waiting` | Nombre de requêtes en attente de prefill |
| `total_requests_processed` | Total des requêtes traitées depuis le démarrage |
| `total_prompt_tokens` | Total des tokens de prompt traités depuis le démarrage |
| `total_completion_tokens` | Total des tokens de complétion générés depuis le démarrage |
| `metal.active_memory_gb` | Mémoire Metal GPU actuellement utilisée (Go) |
| `metal.peak_memory_gb` | Pic d'utilisation de la mémoire Metal GPU (Go) |
| `metal.cache_memory_gb` | Utilisation de la mémoire cache Metal (Go) |
| `cache` | Statistiques du cache (type, entrées, taux de hit, utilisation mémoire) |
| `requests` | Liste des requêtes actives avec détails par requête |

Champs par requête dans `requests` :

| Champ | Description |
|-------|-------------|
| `request_id` | Identifiant unique de la requête |
| `phase` | Phase actuelle : `queued`, `prefill` ou `generation` |
| `tokens_per_second` | Débit de génération pour cette requête |
| `ttft_s` | TTFT (secondes) |
| `progress` | Pourcentage de complétion (0.0 à 1.0) |
| `cache_hit_type` | Type de correspondance dans le cache : `exact`, `prefix`, `supersequence`, `lcp` ou `miss` |
| `cached_tokens` | Nombre de tokens servis depuis le cache |
| `generated_tokens` | Tokens générés jusqu'à présent |
| `max_tokens` | Nombre maximal de tokens demandés |

## Tool Calling

Activez le tool calling compatible OpenAI avec `--enable-auto-tool-choice` :

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Utilisez l'option `--tool-call-parser` pour sélectionner le parser adapté à votre modèle :

| Parser | Modèles |
|--------|---------|
| `auto` | Détection automatique (essaie tous les parsers) |
| `mistral` | Mistral, Devstral |
| `qwen` | Qwen, Qwen3 |
| `llama` | Llama 3.x, 4.x |
| `hermes` | Hermes, NousResearch |
| `deepseek` | DeepSeek V3, R1 |
| `kimi` | Kimi K2, Moonshot |
| `granite` | IBM Granite 3.x, 4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7, GLM-4.7-Flash |

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

Voir le [Guide du tool calling](tool-calling.md) pour la documentation complète.

## Modèles reasoning

Pour les modèles qui exposent leur processus de réflexion (Qwen3, DeepSeek-R1), utilisez `--reasoning-parser` pour séparer le reasoning de la réponse finale :

```bash
# Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

La réponse de l'API inclut un champ `reasoning` avec le processus de réflexion du modèle :

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

En streaming, les fragments de reasoning arrivent en premier, suivis des fragments de contenu :

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

Voir le [Guide des modèles reasoning](reasoning.md) pour tous les détails.

## Sortie structurée (mode JSON)

Forcez le modèle à retourner du JSON valide en utilisant `response_format` :

### Mode JSON Object

Retourne n'importe quel JSON valide :

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### Mode JSON Schema

Retourne du JSON conforme à un schéma spécifique :

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
# Output validated against schema
data = json.loads(response.choices[0].message.content)
assert "colors" in data
```

### Exemple curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## Exemples curl

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuration du streaming

Contrôlez le comportement du streaming avec `--stream-interval` :

| Valeur | Comportement |
|--------|--------------|
| `1` (par défaut) | Envoie chaque token immédiatement |
| `2-5` | Regroupe les tokens avant envoi |
| `10+` | Débit maximal, sortie plus fragmentée |

```bash
# Smooth streaming
vllm-mlx serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

## Intégration Open WebUI

```bash
# 1. Start vllm-mlx server
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# 2. Start Open WebUI
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000
```

## Déploiement en production

### Avec systemd

Créez `/etc/systemd/system/vllm-mlx.service` :

```ini
[Unit]
Description=vLLM-MLX Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching --use-paged-cache --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vllm-mlx
sudo systemctl start vllm-mlx
```

### Paramètres recommandés

Pour une production avec 50 utilisateurs simultanés ou plus :

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
