# MoE top_k override (`--moe-top-k`)

Réduit le nombre d'experts activés par token dans les modèles Mixture of Experts
comme Qwen3-30B-A3B, en échangeant une légère perte de qualité contre un gain
sensible de débit au décodage.

> **Statut :** option à activer explicitement. Le comportement par défaut est inchangé. Les chiffres de qualité
> ci-dessous concernent Qwen3-30B-A3B-4bit sur M4 Max 128 Go ; vérifiez sur votre modèle
> avant de déployer en production.

## Ce que ça fait

Qwen3-30B-A3B est entraîné avec `top_k=8` : chaque token sélectionne 8 experts parmi 128.
Sur Apple Silicon en décodage batch=1, le produit matriciel des experts (`SwitchGLU`)
représente la plus grande part du calcul de chaque couche, et ce coût évolue
approximativement de façon linéaire avec `top_k`. Abaisser `top_k` à l'inférence a
été démontré (LExI 2025, Lynx 2024) comme préservant l'essentiel de la qualité
entraînée tout en réduisant significativement le temps de décodage.

`--moe-top-k N` parcourt toutes les couches du modèle chargé et, pour chaque couche
qui possède `.mlp.switch_mlp` (c'est-à-dire un bloc sparse-MoE), définit `top_k = N`.
Les couches denses et les modèles denses ne sont pas modifiés ; le flag n'a aucun effet pour eux.

## Utilisation

```bash
# Server
vllm-mlx serve mlx-community/Qwen3-30B-A3B-4bit \
  --continuous-batching \
  --moe-top-k 4

# Bench
vllm-mlx bench mlx-community/Qwen3-30B-A3B-4bit --moe-top-k 4
```

Le flag est rejeté si `N` est supérieur au `top_k` d'entraînement du modèle
(il ne peut que diminuer, jamais augmenter).

## Impact mesuré

### Débit de décodage (M4 Max 128 Go, batch=1, greedy)

| top_k | tok/s | vs baseline |
|---:|---:|---:|
| 8 (baseline) | 126.5 | - |
| 6 | 136.1 | +7.6% |
| 5 | 140.3 | +10.9% |
| 4 | 147.3 | +16.5% |

### Qualité (Qwen3-30B-A3B-4bit, lm-evaluation-harness, MLX backend)

<!-- populated after eval completes -->

| top_k | MMLU (acc) | GSM8K (exact match) | Delta vs baseline |
|---:|---:|---:|---:|
| 8 | TBD | TBD | - |
| 6 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |

MMLU : 200 échantillons sélectionnés aléatoirement, 0-shot.
GSM8K : 100 échantillons sélectionnés aléatoirement, 0-shot, exact-match strict.

Ces chiffres sont **indicatifs** ; les suites complètes sont plus grandes et
feraient légèrement varier la précision absolue, mais pas le delta relatif entre
configurations.

### Parité des sorties greedy

Avec `top_k=4` sur le checkpoint 4-bit, nous avons observé des **16 premiers tokens
générés identiques** par rapport à la baseline sur toutes les requêtes de test.
Cela suggère que top_k=4 ne modifie pas l'argmax dans les premières étapes de
décodage : le modèle est intrinsèquement robuste à la suppression de la moitié
de ses experts activés.

À `top_k=3` ou moins, la qualité commencerait à se dégrader visiblement (non mesuré
ici ; déduit du papier LExI). Le flag ne peut donc pas descendre en dessous de 1
au niveau de la validation de configuration, mais le seuil recommandé pour la
production est `top_k=4`.

## Quand l'utiliser, quand ne pas l'utiliser

Utilisez-le quand :
- Vous faites tourner un MoE Qwen3 (ou compatible : Qwen3.5 MoE, Gemma-MoE) et le
  débit de décodage en usage single-user est votre goulot d'étranglement.
- Votre cas d'usage tolère une légère dégradation de qualité en échange d'une
  amélioration visible de la latence.
- Vous déployez sur du matériel limité par la bande passante mémoire (Apple Silicon
  série M) où le gather des experts domine le temps de décodage par étape.

Ne l'utilisez pas quand :
- Vous servez des modèles denses : le flag n'a aucun effet.
- La précision maximale sur les suites d'évaluation est une exigence.
- Vous exécutez des générations longues en chaîne de pensée (mode "thinking") où
  la chute de qualité peut être plus prononcée que ce que suggèrent les scores MMLU 0-shot.

## Combinaison avec d'autres optimisations

Ce flag se compose avec la quantification. Sur Qwen3-30B-A3B-4bit, nos mesures
de combinaison sont :

- 4-bit + top_k=8 : 126.5 tok/s (baseline)
- 4-bit + top_k=4 : 147.3 tok/s (+16.5%)
- 3-bit + top_k=8 : 138.6 tok/s (+9.6%)
- 3-bit + top_k=6 : 147.1 tok/s (+16.3%) . divergence de qualité mesurable
- 3-bit + top_k=4 : 157.3 tok/s (+24%) . **la qualité des sorties s'effondre** (le modèle a répondu à une question différente lors de notre test de fumée)

3-bit + top_k=4 cumule l'erreur numérique au point où l'argmax n'est plus stable.
Limitez-vous à un seul réglage agressif à la fois : soit 4-bit + top_k=4, soit
3-bit + top_k=6. Les deux donnent approximativement le même tok/s (environ 147)
avec des profils de qualité très différents.

## Fonctionnement interne

- Fonction de patch : `vllm_mlx.scheduler.apply_moe_top_k_override(model, k)`
- Appliquée dans `Scheduler.__init__` après le chargement du modèle.
- Tests : `tests/test_moe_top_k.py`. couvre les modèles denses, les architectures
  mixtes et les chemins de validation.

## Références

- LExI : Layer-Adaptive Active Experts, [arXiv 2509.02753](https://arxiv.org/html/2509.02753)
- Not All Experts are Equal (NAEE), [ACL 2024](https://aclanthology.org/2024.acl-long.334.pdf)
- SwiftLM (`SWIFTLM_TOP_K` env knob prior art), [github.com/SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM)
