# Modded-NanoGPT — dépôt d'entraînement

Fork personnel du [NanoGPT speedrun](README_upstream.md), réorganisé pour faire
évoluer et comparer plusieurs versions de la pipeline d'entraînement au fil du
temps.

## Structure

```
├── v1/                  # version 1 de la pipeline (autonome)
│   ├── train_gpt.py     # point d'entrée (main, logging, boucle d'entraînement)
│   ├── dist_setup.py    # setup distribué (rank, device, init_process_group)
│   ├── config.py        # hyperparamètres + schedule d'entraînement
│   ├── fp8.py           # custom ops FP8
│   ├── model.py         # définition du modèle (GPT)
│   ├── optimizers.py    # NorMuon + Adam, Polar Express, sparse comms
│   ├── data.py          # data loader distribué
│   ├── training.py      # TrainingManager (ordonnancement optimizer/schedule)
│   ├── triton_kernels.py
│   ├── export_model.py  # export checkpoint -> safetensors
│   ├── generate.py      # génération depuis un export
│   ├── run.sh           # lancement (torchrun)
│   └── README.md        # notes spécifiques à la version
├── data/                # scripts de téléchargement + dataset FineWeb
├── evals/               # évaluations (hellaswag)
├── logs/                # logs de runs + checkpoints (non versionné)
├── exports/             # exports de modèles (non versionné)
└── (racine)             # bench_kernels.py, perf_model.py, test.py
                         # + train_gpt.py / triton_kernels.py dont ils dépendent
                         # — temporaire, en attendant un dépôt de bench dédié
```

## Versions

Chaque version est un dossier autonome (`v1/`, `v2/`, …) contenant son propre
`train_gpt.py` et les modules dont il dépend (`triton_kernels.py`, …) —
`train_gpt.py` snapshotte le code de tous les `*.py` de son dossier dans le
log, donc ils doivent rester ensemble. Cette autonomie permet de modifier une
version sans affecter les autres, et donc de comparer les approches à
périmètre constant.

Pour créer une nouvelle version :

```bash
cp -r v1 v2
# modifier v2/, ajuster v2/run.sh
```

## Lancer un entraînement

Tout se lance **depuis la racine** du dépôt :

```bash
bash v1/run.sh
```

Variables d'environnement utiles :

- `DATA_PATH` : racine des données (défaut `.`, attend `data/fineweb10B/*.bin`)
- `CKPT_EVERY` : fréquence des checkpoints en steps (`0` = désactivé, défaut 250)
- `CKPT_PATH` : chemin du checkpoint (défaut `logs/ckpt_latest.pt`)
- `RESUME_CKPT` : checkpoint depuis lequel reprendre un run

Les logs de run sont écrits dans `logs/<run_id>.txt` ; comparer deux versions =
comparer leurs logs (loss, temps/step).

## Données

Voir `data/` : `python data/cached_fineweb10B.py` télécharge les shards
FineWeb 10B dans `data/fineweb10B/`.

## Export et génération

```bash
python v1/export_model.py --ckpt logs/ckpt_latest.pt --out exports/mon_modele
python v1/generate.py --model exports/mon_modele --prompt "Once upon a time"
```

## Notes

- Le bench à la racine (`bench_kernels.py`, `perf_model.py`, `test.py`) partira
  dans un dépôt séparé ; les doublons racine de `train_gpt.py` /
  `triton_kernels.py` disparaîtront à ce moment-là (le bench en dépend).
- `export_model.py` et `generate.py` partiront dans un dépôt d'inférence dédié.
- Historique et techniques du speedrun d'origine : voir `README_upstream.md`.
