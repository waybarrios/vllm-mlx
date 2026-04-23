# Warm Prompts

Pré-remplissez le prefix cache au démarrage du serveur afin que la **première** requête
envoyée par un agent trouve un cache déjà chaud, sans payer le coût complet du prefill
pour son system prompt de plusieurs kilo-octets.

## Quand utiliser cette fonctionnalité

Les charges de travail agent. proxies vers des assistants de code ou de raisonnement, serveurs MCP,
orchestrateurs multi-agents. envoient toujours le même system prompt. Aujourd'hui,
la première requête d'un serveur froid paie le prefill complet pour ce system prompt.
Sur un modèle de plusieurs milliards de paramètres, cela représente plusieurs secondes de
TTFT, précisément au moment où un utilisateur attend la première réponse de son nouvel agent.

Si vous connaissez les system prompts de vos agents au moment du déploiement, écrivez-les
dans un fichier JSON et pointez `--warm-prompts` dessus. Le serveur exécute une complétion
de chat avec `max_tokens=1` pour chacun au démarrage, l'état KV cache est chargé dans le
prefix cache, et la première vraie requête correspond via strict-prefix.

Nécessite `--continuous-batching` (le prefix cache y est hébergé).

## Exemple rapide

```bash
# Écrivez une seule fois les agents qui vous intéressent
cat > ~/.config/vllm-mlx/agents.json <<'JSON'
[
  [{"role": "system", "content": "You are a code assistant..."}]
]
JSON

# Pointez le serveur dessus
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json
```

Au démarrage vous verrez :

```
[lifespan] Warm-up done (strict-prefix): 1 completed, 0 skipped,
           1431 prompt tokens in 0.2s
```

La première vraie requête partageant le system prompt réchauffé atteint le cache
avec `tokens_saved` proche de la longueur du prompt de warm-up.

## Format de fichier

Une liste JSON de premier niveau. Chaque entrée est elle-même une liste de messages de chat,
de même forme que `messages` dans `/v1/chat/completions`.

```json
[
  [
    {"role": "system", "content": "You are a code assistant..."}
  ],
  [
    {"role": "system", "content": "You are a senior code reviewer..."}
  ],
  [
    {"role": "system", "content": "You are a planner..."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello, what are we planning?"}
  ]
]
```

Les system prompts à message unique sont le cas le plus courant. Les historiques multi-tours
sont pris en charge pour les scénarios où vous souhaitez réchauffer un début de conversation
spécifique (exemples few-shot, persona d'assistant récurrente).

## Dimensionnement

Les warm prompts sont traités **en parallèle** via `asyncio.gather`, donc N entrées
déclenchent N prefills simultanés au démarrage. Chaque prefill alloue du KV cache
pour la longueur de son prompt.

**Recommandation : 1 à 3 entrées.** Cela couvre les chemins critiques des déploiements
agent typiques (une persona par entrée). Un fichier warm-prompts très grand sur un modèle
à mémoire limitée peut épuiser la marge disponible au démarrage.

Si vous devez réchauffer des dizaines de personas, ouvrez une issue en décrivant votre
charge de travail et nous pourrons ajouter un paramètre `--warm-prompts-concurrency=N`.

## Benchmarks

**Configuration.** M4 Max, 128 Go de mémoire unifiée. Deux serveurs séparés par mesure
(froid et chaud), démarrage à froid isolé. Jeu de prompts `long` (~2 500 tokens utilisateur)
précédé d'un system prompt d'environ 1 700 tokens correspondant au prompt de warm-up.
`max_tokens=128`. bench-serve avec `--skip-preflight-token-count` afin que le preflight
`count_prompt_tokens` ne pollue pas le cache.

| Modèle | conc | TTFT froid | TTFT chaud | Accélération |
|--------|-----:|-----------:|-----------:|-------------:|
| Qwen3-0.6B-8bit | 1 | 563 ms | 419 ms | 1.34x |
| Qwen3-0.6B-8bit | 4 | 1 723 ms | 1 282 ms | 1.34x |
| Qwen3-0.6B-8bit | 8 | 3 708 ms | 2 661 ms | 1.39x |
| Llama-3.2-3B-Instruct-4bit | 1 | 1 754 ms | 1 060 ms | 1.65x |
| Llama-3.2-3B-Instruct-4bit | 4 | 5 926 ms | 3 945 ms | 1.50x |
| Llama-3.2-3B-Instruct-4bit | 8 | 15 161 ms | 9 820 ms | 1.54x |
| Qwen3-4B-4bit | 1 | 4 937 ms | 2 191 ms | 2.25x |
| Qwen3-4B-4bit | 4 | 12 535 ms | 9 623 ms | 1.30x |
| Qwen3-4B-4bit | 8 | 38 148 ms | 23 878 ms | 1.60x |
| Qwen3.6-35B-A3B-4bit (MoE/hybrid) | 1 | 2 400 ms | 1 603 ms | 1.50x |
| Qwen3.6-35B-A3B-4bit | 4 | 8 735 ms | 6 054 ms | 1.44x |
| Qwen3.6-35B-A3B-4bit | 8 | 22 419 ms | 14 409 ms | 1.56x |

Les 12 configurations s'améliorent toutes. Les gains de TTFT sont les plus importants
quand le ratio prompt/total est le plus élevé (conc=1, long system prompt) et restent
significatifs sous charge concurrente.

**La génération tok/s** est neutre (dans ±5 %) pour les modèles denses.
Qwen3.6-35B-A3B (MoE) affiche une baisse de décodage de 20 à 35 % à conc >= 4, qui semble
liée à l'interaction du routage MoE avec la planification en batch. Les gains de TTFT
dominent néanmoins la latence de bout en bout sur les charges agent, mais tenez-en compte
si votre flux de travail est fortement limité par le décodage à forte concurrence.

## Fonctionnement interne

Le warm-up naïf. rendu du template de chat avec un message utilisateur fictif et mise en
cache des tokens. ne fonctionne pas pour les modèles hybrides SSM+attention
(Qwen3.5-MoE, Qwen3.6-MoE). Leurs couches de cache incluent un état SSM qui ne peut pas
être tronqué, si bien que `memory_cache.py` désactive la correspondance LCP. Le contenu
utilisateur fictif diverge du vrai contenu utilisateur, et une entrée mise en cache au
niveau des tokens n'est plus un strict prefix d'aucune vraie requête.

Le warmer ici rend le template de chat **deux fois** avec deux contenus utilisateur
distincts (`"__PROBE_A__"` et `"__PROBE_B__"`), trouve la position de caractère où les deux
chaînes divergent, puis tronque le premier rendu à cette frontière. Cette chaîne tronquée.
tout ce qui précède l'insertion du contenu utilisateur. est ce qui est envoyé au moteur.

Comme le chemin de vraie requête du moteur rend aussi le template avec `tokenize=False` puis
laisse le tokenizer encoder le résultat, les tokens du warm-up sont garantis d'être un strict
prefix de toute vraie requête avec un system prompt correspondant et un historique de chat
vide. Les correspondances strict-prefix fonctionnent sur tous les types de couche de cache,
y compris les chemins hybrides où LCP est désactivé.

## Administration

### Vider le prefix cache en mémoire

```bash
curl -X DELETE http://localhost:8000/v1/cache/prefix
```

Si le serveur a été démarré avec `--warm-prompts`, le warm-up se relance en arrière-plan
après la suppression. La réponse est retournée immédiatement sans attendre la fin du
re-warm.

Réponse :

```json
{"status": "cleared", "rewarm_scheduled": true}
```

### Inspecter l'état du cache

```bash
curl http://localhost:8000/v1/status | jq '.cache'
```

Après le démarrage avec warm-prompts, vous verrez `entry_count > 0` avant la première
requête utilisateur.

## Mesurer l'impact sur votre configuration

Pour mesurer l'impact sur votre modèle et vos prompts, utilisez `bench-serve` :

```bash
# Froid : sans warm-prompts
vllm-mlx serve MODEL --continuous-batching &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag cold \
  --output cold.csv --format csv

# Chaud : même configuration + --warm-prompts
vllm-mlx serve MODEL --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag warm \
  --output warm.csv --format csv
```

`--skip-preflight-token-count` est activé automatiquement quand
`--system-prompt-file` est fourni, afin que le preflight `count_prompt_tokens`
ne pollue pas le cache. Comparez `cold.csv` et `warm.csv` pour votre charge de travail.
