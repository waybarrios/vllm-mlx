# Continuous Batching

Le continuous batching permet d'augmenter le throughput lors du traitement de plusieurs utilisateurs simultanés.

## Activer le Continuous Batching

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching
```

## Avec le Paged Cache

Pour un partage mémoire efficace des préfixes :

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --use-paged-cache
```

## Fonctionnement

### Mode simple (par défaut)
- Une seule requête à la fois
- Throughput maximal pour un utilisateur unique
- Aucune surcharge liée au batching

### Mode Continuous Batching
- Plusieurs requêtes traitées simultanément
- Meilleur throughput pour les utilisateurs concurrents
- Légère surcharge par requête

### Paged Cache
- Le KV cache est stocké en blocs de taille fixe
- Les prompts système identiques partagent les mêmes blocs
- Économies mémoire : 80 % et plus pour 10 utilisateurs simultanés ou davantage

## Résultats de performance

**Résultats du Continuous Batching (M4 Max, 128 Go) :**

| Modèle | Requête unique | Batch (5 req) | Accélération |
|--------|----------------|---------------|--------------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*Le batching de 5 requêtes simultanées améliore le throughput d'un facteur 1,5 à 3.*

## Performance en Streaming

**Performance en streaming (M4 Max, 128 Go) :**

| Modèle | TTFT | Vitesse de génération |
|--------|------|-----------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6 ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7 ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0 ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2 ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1 ms | 140.3 tok/s |

*TTFT = Time to First Token*

## Configuration du Streaming

Contrôlez la cadence d'envoi des tokens avec `--stream-interval` :

```bash
# Chaque token (le plus fluide)
vllm-mlx serve model --continuous-batching --stream-interval 1

# Tokens groupés (préférable pour les connexions à latence élevée)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

| Valeur | Comportement |
|--------|-------------|
| `1` | Envoie chaque token immédiatement |
| `2-5` | Regroupe les tokens avant l'envoi |
| `10+` | Throughput maximal, sortie plus fragmentée |

## Gestion de la mémoire

Pour les grands modèles, le prefix cache peut consommer une quantité significative de mémoire. Le cache adaptatif la gère automatiquement :

```bash
# Détection automatique (utilise 20 % de la RAM disponible)
vllm-mlx serve model --continuous-batching

# Limite explicite
vllm-mlx serve model --continuous-batching --cache-memory-mb 2048

# Pourcentage personnalisé
vllm-mlx serve model --continuous-batching --cache-memory-percent 0.10
```

| Option | Description |
|--------|-------------|
| `--cache-memory-mb` | Définit une limite explicite en Mo |
| `--cache-memory-percent` | Fraction de la RAM disponible (par défaut : 0,20) |
| `--no-memory-aware-cache` | Utilise le cache historique basé sur le nombre d'entrées |

## Prefix Cache

Le prefix caching réutilise le KV cache pour les prompts répétés.

### Fonctionnement

```
User 1: System prompt (500 tokens) → Creates 8 blocks
User 2: Same system prompt → Shares 8 blocks (ref_count++)
User N: Same system prompt → Shares 8 blocks (ref_count++)

Memory savings: 80%+ for 10+ concurrent users
```

### Stratégie de clé de cache

- **LLM** : `hash(prompt)`
- **Images** : `hash(image_content) + hash(prompt)`
- **Vidéos** : `hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

### Tester le Prefix Cache

```bash
python tests/test_prefix_cache.py
```

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | PASS
  1b     | Same prompt         | HIT      | HIT    | PASS
  1c     | Different prompt    | MISS     | MISS   | PASS
  1d     | Return to prompt 1  | HIT      | HIT    | PASS
======================================================================
```

## Exécuter les benchmarks

```bash
# Benchmark du continuous batching
python tests/test_continuous_batching.py

# Test du prefix cache
python tests/test_prefix_cache.py
```

## Quand l'utiliser

| Scénario | Mode |
|----------|------|
| Utilisateur unique, vitesse maximale | Simple (par défaut) |
| Plusieurs utilisateurs simultanés | `--continuous-batching` |
| Grands modèles (7B et plus) | `--continuous-batching --cache-memory-mb 2048` |
| Production avec prompts partagés | `--continuous-batching --use-paged-cache` |

## Configuration en production

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000
```
