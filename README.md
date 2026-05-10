# DCL (decentralized version)

Decentralized Differentiable Commitment Learning: each agent maintains its own estimation of the co-player’s networks.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Hyperparameters are in `[configs/config.yaml](configs/config.yaml)` (Hydra). Run:

```bash
python main.py
```

Override any field: `python main.py game=IPC mega_step=2 max_steps=16 batch_size=512`.

Extra keys vs the centralized `main` branch:

- `epsilon`, `epsilon_decay` — exploration schedule in the repeated-game rollout / updates  
- `perturb` — numerical floor on probabilities

Set `wandb.mode` to `online` or `offline` under `wandb:` when logging.

## Layout

- `[trainer/base_trainer.py](trainer/base_trainer.py)` — shared mega-step encoding + diagnostics helpers for decentralized trainers  
- `[trainer/dcl_repeated_games.py](trainer/dcl_repeated_games.py)`, `[trainer/dcl_grid_game.py](trainer/dcl_grid_game.py)` — game trainers  
- `[utility/agents/nets.py](utility/agents/nets.py)` — shared `SoftmaxNet` / `CriticNet` modules

