# v1

Première version de la pipeline, dérivée du record courant du speedrun
modded-nanogpt (voir `../README_upstream.md`), adaptée pour tourner sur une
seule GPU grand public (RTX 4060 Ti) au lieu de 8×H100.

## Contenu

- `train_gpt.py` : point d'entrée (`main()` : logging, construction du modèle,
  warmup, boucle d'entraînement/validation, checkpoints). Au démarrage, il
  snapshotte dans le log le code de tous les `*.py` du dossier : ils doivent
  donc rester ensemble.
- `dist_setup.py` : setup distribué (rank, world_size, device, accumulation de
  gradient, `init_process_group`).
- `config.py` : hyperparamètres (`Hyperparameters`/`args`) et schedule
  d'entraînement (`TrainingStage`, `TrainingSchedule`, `TRAINING_STAGES`,
  `get_muon_momentum`).
- `fp8.py` : custom ops FP8 (`nanogpt::mm_t` + backward).
- `model.py` : définition du modèle (`GPT`, attention, YaRN, MLP ReLU²,
  `CastedLinearT`, `next_multiple_of_n`).
- `optimizers.py` : optimizer combiné NorMuon + Adam, Polar Express,
  communications sparse.
- `data.py` : data loader distribué (shards FineWeb, hash bigram,
  `DistributedDataGenerator`).
- `training.py` : `TrainingManager` (ordonnancement des reduces/gathers,
  schedule, split embed/lm_head).
- `triton_kernels.py` : kernels Triton fusionnés (linear+ReLU², cross-entropy
  softcapée FP8, etc.).
- `export_model.py` : exporte un checkpoint vers un dossier d'inférence
  (`model.safetensors` + `config.json`) en défaisant les « banks » de poids
  optimisées pour l'optimizer et en recalculant les tables YaRN.
- `generate.py` : génération de texte depuis un dossier exporté.
- `run.sh` : lancement depuis la racine du dépôt.

## Adaptations locales par rapport au speedrun upstream

- `torchrun --nproc_per_node=1` (mono-GPU) dans `run.sh`.
- Reprise sur checkpoint : `RESUME_CKPT=<fichier>` recharge modèle, optimizer
  et état du data loader (voir le patch « RTX 4060 Ti » en fin de
  `train_gpt.py`).
- Checkpoints périodiques : `CKPT_EVERY` (défaut 250 steps, `0` = désactivé),
  `CKPT_PATH` (défaut `logs/ckpt_latest.pt`).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` pour limiter la fragmentation
  mémoire.

## Usage

Depuis la racine du dépôt :

```bash
bash v1/run.sh                                          # entraînement
RESUME_CKPT=logs/ckpt_latest.pt bash v1/run.sh          # reprise
python v1/export_model.py --ckpt logs/ckpt_latest.pt --out exports/mon_modele
python v1/generate.py --model exports/mon_modele -i     # génération interactive
```
