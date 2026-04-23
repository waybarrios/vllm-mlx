# Reasoning Models

vllm-mlx prend en charge les reasoning models qui affichent leur processus de thinking avant de fournir une réponse. Des modèles comme Qwen3 et DeepSeek-R1 encapsulent leur reasoning dans des balises `<think>...</think>`, et vllm-mlx peut analyser ces balises pour séparer le reasoning de la réponse finale.

## Pourquoi utiliser le reasoning parsing ?

Lorsqu'un reasoning model génère une sortie, elle ressemble généralement à ceci :

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

Sans reasoning parsing, vous obtenez la sortie brute avec les balises incluses. Avec le reasoning parsing activé, le processus de thinking et la réponse finale sont séparés dans des champs distincts de la réponse de l'API.

## Démarrage rapide

### Démarrer le serveur avec un reasoning parser

```bash
# For Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### Format de réponse de l'API

Lorsque le reasoning parsing est activé, la réponse de l'API inclut un champ `reasoning` :

**Réponse sans streaming :**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

**Réponse en streaming :**

Les fragments sont envoyés séparément pour le reasoning et le contenu. Pendant la phase de reasoning, les fragments ont le champ `reasoning` renseigné. Lorsque le modèle passe à la réponse finale, les fragments ont le champ `content` renseigné :

```json
{"delta": {"reasoning": "Let me analyze"}}
{"delta": {"reasoning": " this step by step."}}
{"delta": {"reasoning": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

## Utilisation avec le SDK OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What are the prime numbers less than 10?"}]
)

message = response.choices[0].message
print("Reasoning:", message.reasoning)  # The thinking process
print("Answer:", message.content)        # The final answer
```

### Streaming avec Reasoning

```python
reasoning_text = ""
content_text = ""

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2 + 2 = ?"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning') and delta.reasoning:
        reasoning_text += delta.reasoning
        print(f"[Thinking] {delta.reasoning}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## Parsers disponibles

### Parser Qwen3 (`qwen3`)

Pour les modèles Qwen3 qui utilisent explicitement les balises `<think>` et `</think>`.

- Nécessite **les deux** balises ouvrante et fermante
- Si les balises sont absentes, la sortie est traitée comme du contenu ordinaire
- Recommandé pour : Qwen3-0.6B, Qwen3-4B, Qwen3-8B et les modèles similaires

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### Parser DeepSeek-R1 (`deepseek_r1`)

Pour les modèles DeepSeek-R1 qui peuvent omettre la balise ouvrante `<think>`.

- Plus permissif que le parser Qwen3
- Gère les cas où `<think>` est implicite
- Le contenu avant `</think>` est traité comme du reasoning même en l'absence de `<think>`

```bash
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

## Fonctionnement

Le reasoning parser utilise une détection textuelle pour identifier les balises de thinking dans la sortie du modèle. Pendant le streaming, il suit la position courante dans la sortie afin d'acheminer chaque token vers `reasoning` ou `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

L'analyse est sans état et s'appuie sur le texte accumulé pour déterminer le contexte, ce qui la rend robuste dans les scénarios de streaming où les tokens peuvent arriver en fragments arbitraires.

## Conseils pour de meilleurs résultats

### Rédaction des prompts

Les reasoning models fonctionnent mieux lorsque vous encouragez une réflexion étape par étape :

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### Gestion de l'absence de reasoning

Certains prompts peuvent ne pas déclencher de reasoning. Dans ce cas, `reasoning` vaut `None` et toute la sortie va dans `content` :

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### Température et reasoning

Les températures basses tendent à produire des schémas de reasoning plus cohérents :

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## Compatibilité ascendante

Lorsque `--reasoning-parser` n'est pas spécifié, le serveur se comporte comme avant :
- Les balises de thinking sont incluses dans le champ `content`
- Aucun champ `reasoning` n'est ajouté aux réponses

Cela garantit que les applications existantes continuent de fonctionner sans modification.

## Exemple : résolveur de problèmes mathématiques

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def solve_math(problem: str) -> dict:
    """Solve a math problem and return reasoning + answer."""
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a math tutor. Show your work."},
            {"role": "user", "content": problem}
        ],
        temperature=0.2
    )

    message = response.choices[0].message
    return {
        "problem": problem,
        "work": message.reasoning,
        "answer": message.content
    }

result = solve_math("If a train travels 120 km in 2 hours, what is its average speed?")
print(f"Problem: {result['problem']}")
print(f"\nWork shown:\n{result['work']}")
print(f"\nFinal answer: {result['answer']}")
```

## Exemples avec curl

### Sans streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Avec streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## Résolution de problèmes

### Champ reasoning absent de la réponse

- Vérifiez que le serveur a bien été démarré avec `--reasoning-parser`
- Vérifiez que le modèle utilise effectivement des balises de thinking (tous les prompts ne déclenchent pas le reasoning)

### Le reasoning apparaît dans le contenu

- Le modèle n'utilise peut-être pas le format de balises attendu
- Essayez un autre parser (`qwen3` ou `deepseek_r1`)

### Reasoning tronqué

- Augmentez `--max-tokens` si le modèle atteint la limite de tokens en plein milieu de sa réflexion

## Voir aussi

- [Modèles pris en charge](../reference/models.md) - Modèles qui prennent en charge le reasoning
- [Configuration du serveur](server.md) - Toutes les options du serveur
- [Référence CLI](../reference/cli.md) - Options de la ligne de commande
