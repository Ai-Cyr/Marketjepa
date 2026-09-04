# MarketJEPA

Fondation d'une **Joint-Embedding Predictive Architecture (JEPA)** appliquée aux séries
temporelles de marché. Le modèle apprend, de manière auto-supervisée, à prédire la
*représentation latente* de segments futurs (ou masqués) d'une série à partir du contexte
visible, sans jamais reconstruire les prix bruts.

L'objectif est d'obtenir un encodeur générique de dynamique de marché, réutilisable en aval
(prévision, détection de régime, classification, allocation).

## Principe

```
fenêtre (T pas, F features)
        │
        ├─► patchs ─► encodeur cible (EMA, sans gradient) ─► h_cible  ──┐
        │                                                                 │  smooth-L1
        └─► patchs de contexte ─► encodeur de contexte ─► prédicteur ─► ĥ ─┘
```

- **Encodeur de contexte** : Transformer entraîné par rétropropagation, ne voit que les patchs
  de contexte.
- **Encodeur cible** : copie de l'encodeur de contexte mise à jour par moyenne mobile
  exponentielle (EMA). Il encode la fenêtre entière et fournit les cibles.
- **Prédicteur** : petit Transformer qui, à partir des représentations de contexte et de la
  position des patchs cibles, prédit leurs représentations.
- **Masquage** : `future` (contexte = passé, cible = futur, adapté à la prévision) ou
  `block` (blocs contigus aléatoires, style I-JEPA).

Aucune reconstruction au niveau du signal : la perte est calculée dans l'espace latent,
ce qui évite de gaspiller de la capacité sur le bruit haute fréquence des prix.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Démarrage rapide

Entraînement sur données synthétiques (GBM à changement de régime) :

```bash
marketjepa-train --config configs/default.yaml --synthetic-steps 20000 --epochs 2
```

Entraînement sur vos propres CSV OHLCV (une colonne date en index, colonnes
`open, high, low, close, volume`), un fichier par actif :

```bash
marketjepa-train --config configs/default.yaml --data data/BTC.csv data/ETH.csv
```

Le checkpoint (`checkpoints/marketjepa.pt`) contient les poids, l'optimiseur, la configuration
et les statistiques de normalisation nécessaires à l'inférence.

Utilisation en Python :

```python
import torch
from marketjepa import MarketJEPA, load_config
from marketjepa.data import build_dataset, generate_synthetic_ohlcv

cfg = load_config("configs/default.yaml")
dataset, stats = build_dataset(generate_synthetic_ohlcv(5000), cfg.data)
model = MarketJEPA(num_features=cfg.data.num_features, cfg=cfg.model)

window = dataset[0].unsqueeze(0)          # (1, T, F)
patch_repr = model.encode(window)         # (1, N, d_model)
window_repr = model.embed(window)         # (1, d_model)
```

## Structure du projet

```
src/marketjepa/
├── config.py            # dataclasses de configuration + chargement YAML
├── cli.py               # point d'entrée `marketjepa-train`
├── data/
│   ├── features.py      # OHLCV -> features log-ratio stationnaires, z-score
│   ├── synthetic.py     # générateur OHLCV synthétique (régimes markoviens)
│   └── dataset.py       # fenêtres glissantes multi-actifs, pipeline complet
├── models/
│   ├── embedding.py     # patchs -> tokens, encodage positionnel
│   ├── encoder.py       # encodeur Transformer
│   ├── predictor.py     # prédicteur latent avec mask tokens
│   └── jepa.py          # MarketJEPA (contexte + cible EMA + prédicteur)
├── training/
│   ├── masking.py       # stratégies de masquage (future, block)
│   └── trainer.py       # boucle d'entraînement, schedules, checkpoints
└── utils/               # seed, device
configs/default.yaml     # configuration de référence
tests/                   # suite pytest
docs/architecture.md     # notes de conception
```

## Features d'entrée

| Feature              | Définition            |
|----------------------|-----------------------|
| `log_return`         | log(C_t / C_{t-1})    |
| `log_range`          | log(H_t / L_t)        |
| `log_body`           | log(C_t / O_t)        |
| `log_gap`            | log(O_t / C_{t-1})    |
| `log_volume_change`  | log(V_t / V_{t-1})    |

Toutes sont indépendantes du niveau de prix, puis normalisées par z-score ajusté sur le jeu
d'entraînement.

## Développement

```bash
make test     # pytest
make lint     # ruff
make format   # ruff format
```

## Feuille de route

- [ ] Chargeurs de données réels (Parquet, API d'échange) et split temporel train/val
- [ ] Évaluation aval : sonde linéaire sur le signe du rendement futur, détection de régime
- [ ] Masquage multi-cibles et masquage inter-actifs
- [ ] Régularisation anti-effondrement (VICReg) en option
- [ ] Suivi d'expériences (TensorBoard / W&B)
