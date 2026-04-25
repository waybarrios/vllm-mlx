# Benchmarks LLM

## Lancer les benchmarks LLM

```bash
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

## Résultats (M4 Max, 128 Go)

| Modèle | Vitesse de génération | TTFT* | Mémoire |
|--------|-----------------------|-------|---------|
| Qwen3-0.6B-8bit | 402,3 tok/s | 58,6 ms | 0,68 Go |
| Llama-3.2-1B-Instruct-4bit | 463,6 tok/s | 49,2 ms | 0,69 Go |
| Qwen2.5-1.5B-Instruct-4bit | 308,5 tok/s | 86,2 ms | 0,84 Go |
| Llama-3.2-3B-Instruct-4bit | 200,1 tok/s | 81,4 ms | 1,79 Go |
| Qwen3-30B-A3B-4bit | 123,9 tok/s | 126,9 ms | 16,05 Go |
| NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit | 122,9 tok/s | 72,3 ms | 23,98 Go |

*TTFT = Time to First Token (latence jusqu'au premier token généré)

## Résultats (M1 Max, 64 Go)

| Modèle | Requêtes | Tok. prompt | Tok. générés | Temps total (s) | TTFT moyen (ms) | TPOT moyen (ms) | Vitesse génération (tok/s) | Débit total (tok/s) |
|--------|----------|-------------|--------------|-----------------|-----------------|-----------------|---------------------------|---------------------|
| Qwen3-0.6B-8bit | 5 | 56 | 1280 | 5,66 | 119,0 | 3,97 | 251,9 | 236,1 |

## Résultats du continuous batching

| Modèle | Requête unique | Batch (5 req) | Accélération |
|--------|----------------|---------------|--------------|
| Llama-3.2-1B-Instruct-4bit | 299,1 tok/s | 613,0 tok/s | **2,05x** |
| Llama-3.2-3B-Instruct-4bit | 137,6 tok/s | 208,1 tok/s | **1,51x** |
| Qwen3-0.6B-8bit | 328,1 tok/s | 1111,8 tok/s | **3,39x** |
| Qwen3-30B-A3B-4bit | 98,1 tok/s | 233,3 tok/s | **2,38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196,9 tok/s | 322,2 tok/s | **1,64x** |

*Le batching de 5 requêtes simultanées apporte une amélioration du throughput de 1,5 à 3x.*

### Continuous batching (M1 Max, 64 Go)

| Requêtes | Tokens totaux | Temps total (s) | Throughput (tok/s) | Requêtes/sec |
|----------|---------------|-----------------|--------------------|--------------|
| 5 | 315 | 0,64 | 492,5 | 7,82 |

## Performances en streaming

| Modèle | TTFT | Vitesse de génération |
|--------|------|-----------------------|
| Llama-3.2-1B-Instruct-4bit | ~4,6 ms | 218,9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10,7 ms | 93,6 tok/s |
| Qwen3-0.6B-8bit | ~3,0 ms | 328,5 tok/s |
| Qwen3-30B-A3B-4bit | ~10,2 ms | 98,4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7,1 ms | 140,3 tok/s |

### Détokeniseur en streaming (M1 Max, 64 Go)

`vllm-mlx bench-detok` :

| Tokens | Itérations | Temps naïf | Temps streaming | Accélération |
|--------|------------|------------|-----------------|--------------|
| 742 | 5 | 1,69 ms | 0,71 ms | 2,39x |

`examples/benchmark_detokenizer.py` :

| Séquence | Tokens | decode() | Streaming | Accélération |
|----------|--------|----------|-----------|--------------|
| Courte | 8 | 0,029 ms | 0,028 ms | 1,04x |
| Moyenne | 103 | 0,206 ms | 0,129 ms | 1,59x |
| Longue | 511 | 1,040 ms | 0,502 ms | 2,07x |
| 1K | 1191 | 2,446 ms | 1,178 ms | 2,08x |
| 2K | 2381 | 4,949 ms | 2,356 ms | 2,10x |
| 4K | 4761 | 9,887 ms | 5,398 ms | 1,83x |

Accélération moyenne : 1,79x

## Résultats du prefix cache

### Prefix cache (M4 Max, 128 Go)

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | ✓
  1b     | Same prompt         | HIT      | HIT    | ✓
  1c     | Different prompt    | MISS     | MISS   | ✓
  1d     | Return to prompt 1  | HIT      | HIT    | ✓
======================================================================
```

### Prefix cache (M1 Max, 64 Go)

| Test | Attendu | Réel | Temps | Statut |
|------|---------|------|-------|--------|
| Première requête | MISS | MISS | 203,5 ms | PASS |
| Même prompt | HIT | HIT | 131,6 ms | PASS |
| Prompt différent | MISS ou PREFIX_HIT | PREFIX_HIT (5 tok) | 135,3 ms | PASS |

Statistiques finales du cache :

| Hits cache | Misses cache | Taux de hit | Tokens économisés | Accélération avec cache |
|------------|--------------|-------------|-------------------|------------------------|
| 2 | 1 | 66,7 % | 20 | 1,55x |

## Résultats du paged cache

*Test : 20 requêtes d'inférence réelles en 2 rounds avec un prompt système partagé d'environ 286 tokens*

```
======================================================================
  PAGED KV CACHE - REAL INFERENCE TEST
======================================================================

--------------------------------------------------
Test 1: WITHOUT Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.47s
  Throughput: 681.2 tok/s
  Cache hits: 0
  Tokens saved: 0

--------------------------------------------------
Test 2: WITH Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.31s
  Throughput: 765.8 tok/s

  Paged Cache Stats:
    Blocks allocated: 25
    Shared blocks: 4
    Cache hits: 10
    Tokens saved: 2560

==================================================
SUMMARY
==================================================
  Without paged cache: 681.2 tok/s
  With paged cache:    765.8 tok/s

  Speedup: 1.12x
  Cache hits: 10 (all Round 2 requests)
  Tokens saved: 2,560 (~256 tokens × 10 requests)
==================================================
```

### KV cache paginé (M1 Max, 64 Go)

Benchmark d'inférence (20 requêtes) :

| Mode | Temps (s) | Throughput (tok/s) |
|------|-----------|--------------------|
| Sans paged cache | 3,43 | 291,8 |
| Avec paged cache | 3,42 | 292,2 |

| Accélération | Blocs alloués | Blocs partagés | Hits cache | Tokens économisés |
|--------------|---------------|----------------|------------|-------------------|
| 1,00x | 45 | 4 | 10 | 2560 |

Inférence concurrente réelle (20 requêtes) :

| Mode | Temps (s) | Throughput (tok/s) |
|------|-----------|--------------------|
| Sans paged cache | 4,32 | 231,7 |
| Avec paged cache | 4,35 | 229,7 |

| Accélération | Blocs alloués | Blocs partagés | Hits cache | Tokens économisés |
|--------------|---------------|----------------|------------|-------------------|
| 0,99x | 49 | 8 | 10 | 5120 |

Démonstration des économies mémoire :

| Scénario | Économies mémoire |
|----------|-------------------|
| Prompts système partagés | 70,8 % |
| Efficacité mémoire concurrente | 83,5 % |
| Branches avec partage de préfixe | 38,5 % |

## Analyse du détokeniseur en streaming

*Investigation phase 9.1 : `BPEStreamingDetokenizer` de mlx-lm vs `tokenizer.decode()` naïf*

### Contexte

L'approche naïve appelle `decode([token])` pour chaque token. En théorie, les détokeniseurs en streaming offrent une complexité O(T) contre O(T²) pour le décodage naïf.

### Résultats du benchmark isolé

```bash
vllm-mlx bench-detok
```

En réutilisant la même instance de détokeniseur (avec `reset()` entre les utilisations) :

| Séquence | Tokens | decode() naïf | Streaming | Accélération |
|----------|--------|---------------|-----------|--------------|
| Courte | 8 | 0,020 ms | 0,019 ms | 1,05x |
| Moyenne | 103 | 0,155 ms | 0,097 ms | 1,59x |
| Longue | 511 | 0,752 ms | 0,371 ms | **2,03x** |
| 1K tokens | 1191 | 1,743 ms | 0,833 ms | **2,09x** |
| 2K tokens | 2381 | 3,493 ms | 1,737 ms | **2,01x** |

### Constat critique : surcoût de création d'instance

La création d'une nouvelle instance de `BPEStreamingDetokenizer` est **extrêmement coûteuse** :

```
100 tokenizer.detokenizer calls: 5.266s (52.7ms each!)
```

Cela signifie que créer un nouveau détokeniseur par requête ajoute **environ 52 ms de surcoût**, annulant tout bénéfice.

### Impact en conditions réelles

Intégré dans le scheduler (un détokeniseur par requête) :

| Métrique | decode() naïf | Streaming (nouvelle instance) |
|----------|---------------|-------------------------------|
| Throughput (20 req) | 681 tok/s | 275 tok/s |
| Impact | - | **-60 % plus lent** |

### Conclusion

Le détokeniseur en streaming n'est **pas viable actuellement** pour un usage par requête, en raison du coût de création d'instance. L'approche naïve `decode([token])` reste plus rapide en pratique.

**Optimisation future** : pré-créer un pool d'instances de détokeniseur au démarrage et les réutiliser entre les requêtes.

## Référence des métriques

| Métrique | Description |
|----------|-------------|
| **TTFT** | Time to First Token - latence jusqu'à ce que le modèle commence à répondre (ms) |
| **TPOT** | Time Per Output Token - temps entre chaque token généré (ms/token) |
| **Generation TPS** | Tokens de sortie par seconde (tok/s) |
| **Processing TPS** | Tokens d'entrée/prompt traités par seconde (tok/s) |
| **End-to-End Latency** | Temps total de la requête à la réponse complète |
| **Total Throughput** | Tokens totaux (entrée + sortie) par seconde |

## Lancer les benchmarks

```bash
# Benchmark de base
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Avec davantage de prompts
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --prompts 10

# Sauvegarder les résultats
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json

# Test de continuous batching
python tests/test_continuous_batching.py

# Test de prefix cache
python tests/test_prefix_cache.py

# Test de paged cache
python tests/test_paged_cache_real_inference.py

# Benchmark du détokeniseur en streaming
vllm-mlx bench-detok
vllm-mlx bench-detok mlx-community/Llama-3.2-1B-Instruct-4bit --iterations 5
```
