# Notes d'architecture

## Pourquoi une JEPA pour les marchés ?

Les approches génératives (reconstruction de prix, prévision point à point) forcent le modèle à
modéliser le bruit haute fréquence, largement imprévisible. Une JEPA prédit dans l'espace des
représentations : l'encodeur est libre d'ignorer ce qui n'est pas prévisible et de ne conserver
que la structure (régimes de volatilité, tendances, saisonnalités, dépendances volume/prix).

## Flux de données

1. **OHLCV → features** (`data/features.py`) : cinq log-ratios stationnaires. La première ligne
   est perdue (différenciation).
2. **Normalisation** : z-score par feature, statistiques ajustées sur l'entraînement uniquement
   et sauvegardées dans le checkpoint.
3. **Fenêtres** (`data/dataset.py`) : fenêtre de `context_len + target_len` pas, avec `stride`.
   Les fenêtres ne traversent jamais deux actifs.
4. **Patchs** (`models/embedding.py`) : `patch_len` pas consécutifs sont aplatis et projetés en
   un token de dimension `d_model`. Encodage positionnel sinusoïdal.

## Modèle

| Composant          | Rôle                                                          | Gradient |
|--------------------|---------------------------------------------------------------|----------|
| `context_encoder`  | Encode uniquement les patchs de contexte                       | oui      |
| `target_encoder`   | Encode la fenêtre entière ; cibles normalisées par LayerNorm   | non (EMA)|
| `predictor`        | Contexte + mask tokens positionnés → représentations cibles    | oui      |

Perte : smooth-L1 entre prédictions et cibles. Les cibles sont normalisées par LayerNorm
(comme dans I-JEPA) pour stabiliser l'échelle.

### Anti-effondrement

L'asymétrie encodeur de contexte / encodeur cible EMA, combinée à l'arrêt de gradient sur
les cibles, empêche la solution triviale (représentation constante). La montée progressive du
momentum EMA (`ema_momentum_start → ema_momentum_end`) suit la recette I-JEPA/BYOL.

## Masquage

- `future` : contexte = `num_context_patches` premiers patchs, cible = derniers patchs. Le
  modèle apprend une représentation « prédictive du futur », directement pertinente pour la
  prévision.
- `block` : `num_target_blocks` blocs contigus tirés au hasard deviennent la cible, le reste
  est contexte. Le masque est partagé par tout le batch (comme le collator I-JEPA).

Les indices de masque sont des tenseurs 1-D partagés par le batch ; `gather_tokens` accepte
aussi des indices 2-D par échantillon si une stratégie future en a besoin.

## Entraînement

- AdamW, warmup linéaire puis décroissance cosinus.
- Clipping de gradient à 1.0.
- Mise à jour EMA après chaque pas d'optimisation.
- Checkpoint : poids, optimiseur, configuration, historique, statistiques z-score.

## Extensions envisagées

- Tokens d'actif (embedding par symbole) pour l'entraînement multi-actifs.
- Masquage inter-actifs : prédire un actif à partir des autres sur la même période.
- Régularisation VICReg optionnelle sur les représentations de contexte.
- Tête d'évaluation aval (sonde linéaire) et protocole de validation temporelle.
