# Differentiable Commitment Learning
This repository implements DCL ([AISTATS 2025](https://virtual.aistats.org/Conferences/2025)) in our paper [Learning to Negotiate via Voluntary  Commitment](https://arxiv.org/abs/2503.03866).

![Markov Commitment Game (MCG)](https://shuhui-zhu.github.io/images/MCG.jpg)

## Set Up

* Clone this repository.

* Set up the virtual environment.

  ```
  conda create -n virtual_env python=3.10.9
  conda activate virtual_env
  pip install -r requirements.txt
  ```

## Navigation

The implementation of DCL is in the `main` branch; decentralized DCL is in the `decentralized_dcl` branch.

* `env/` — Prisoner's Dilemma, Repeated Purely Conflicting Game, and Grid Game.
* `trainer/` — DCL trainers.
  * `base_trainer.py` — `BaseDCLTrainer` with the shared episode loop, replay-buffer setup, and per-batch update logic.
  * `dcl_repeated_games.py` / `dcl_grid_game.py` — thin subclasses that build the env / agents and add game-specific wandb diagnostics.
* `utility/`
  * `buffer_class.py` — replay buffer.
  * `agents/nets.py` — shared `SoftmaxNet` and `CriticNet` MLPs.
  * `agents/base_agent.py` — `DCL_Agent` with all algorithmic logic (critic / unconstrained / commitment / proposal updates).
  * `agents/agent_class_repeated_games.py` and `agents/agent_class_grid_game.py` — thin game-specific subclasses.
* `configs/config.yaml` — Hydra config holding all hyperparameters (game, model, training, logging).
* `main.py` — Hydra entry point that wires the config to the trainers.

## Running Experiments

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml) and are loaded with [Hydra](https://hydra.cc). Edit that file to set the game, model, and training options, then run:

```
python main.py
```

Use `with_constraints: true` for DCL-IC or `with_constraints: false` for plain DCL. To reproduce the settings in the paper [Learning to Negotiate via Voluntary Commitment](https://arxiv.org/abs/2503.03866) (Appendix D), use the values below for each game:

| Game | `game` | `max_steps` | `mega_step` | `batch_size` | `entropy_coeff` | `temperature` |
|---|---|---:|---:|---:|---:|---:|
| Prisoner's Dilemma | `IPD` | 1 | 1 | 128 | 1.0 | 10.0 |
| Repeated Purely Conflicting Game | `IPC` | 16 | 2 | 512 | 2.0 | 1.0 |
| Grid Game | `Grid` | 16 | 1 | 512 | 2.0 | 1.0 |

If you'd rather not edit the file, any field can also be overridden on the command line using Hydra's `field=value` syntax (note `=`, not `--`):

```
python main.py game=IPC with_constraints=true max_steps=16 mega_step=2 batch_size=512 entropy_coeff=2.0 temperature=1.0
```

> By default Hydra creates a per-run output directory under `outputs/YYYY-MM-DD/HH-MM-SS/`. Add `hydra.run.dir=.` to the command to keep runs in the project root.

## Citation

If you want to cite our paper or this repository, please use the following citation:

```
@inproceedings{zhu2025learning,
  title={Learning to Negotiate via Voluntary Commitment},
  author={Zhu, Shuhui and Wang, Baoxiang and Subramanian, Sriram Ganapathi and Poupart, Pascal},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1459--1467},
  year={2025},
  organization={PMLR}
}
```

## License
See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT.
