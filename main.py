
from env.repeated_game import IteratedPrisonersDilemma, IteratedPureConflict
from env.grid_game import GridSocialDilemmaEnv

from utility.agents.agent_class_repeated_games import DCL_Agent_Repeated_Games
from utility.agents.agent_class_grid_game import DCL_Agent_Grid_Game

from trainer.dcl_repeated_games import Repeated_Game_Trainer
from trainer.dcl_grid_game import Grid_Game_Trainer

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os

os.environ['WANDB_MODE'] = 'disabled'

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ['WANDB_MODE'] = cfg.wandb.mode

    total_steps = cfg.N_iterations * cfg.batch_size
    N_episodes = int(total_steps / cfg.max_steps)

    assert cfg.batch_size % cfg.max_steps == 0  # batch_size should be divisible by max_steps

    wandb.init(
        project=cfg.wandb.project,
        group=cfg.game,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    if cfg.game == 'IPD':
        environment = IteratedPrisonersDilemma
    elif cfg.game == 'IPC':
        environment = IteratedPureConflict
    elif cfg.game == 'Grid':
        environment = GridSocialDilemmaEnv
    else:
        raise ValueError("Invalid game name")

    if cfg.game == 'IPD' or cfg.game == 'IPC':
        trainer = Repeated_Game_Trainer(
            env=environment, agent_class=DCL_Agent_Repeated_Games, mega_step=cfg.mega_step,
            gamma=cfg.gamma, max_steps=cfg.max_steps, temperature=cfg.temperature,
            temperature_decay=cfg.temperature_decay, hidden_dim=cfg.hidden_dim,
            lr_critic=cfg.lr_critic, lr_actor=cfg.lr_actor,
            with_constraints=cfg.with_constraints, buffer_length=cfg.batch_size,
            num_iter_per_batch=cfg.num_iter_per_batch, batch_size=cfg.batch_size,
            N_agents=2, N_episodes=N_episodes, is_entropy=cfg.is_entropy,
            entropy_coeff=cfg.entropy_coeff, entropy_coeff_decay=cfg.entropy_coeff_decay,
        )
    elif cfg.game == 'Grid':
        trainer = Grid_Game_Trainer(
            env=environment, agent_class=DCL_Agent_Grid_Game, gamma=cfg.gamma,
            max_steps=cfg.max_steps, temperature=cfg.temperature,
            temperature_decay=cfg.temperature_decay, hidden_dim=cfg.hidden_dim,
            lr_critic=cfg.lr_critic, lr_actor=cfg.lr_actor,
            with_constraints=cfg.with_constraints, buffer_length=cfg.batch_size,
            num_iter_per_batch=cfg.num_iter_per_batch, batch_size=cfg.batch_size,
            N_agents=2, N_episodes=N_episodes, grid_size=cfg.grid_size,
            is_entropy=cfg.is_entropy, entropy_coeff=cfg.entropy_coeff,
            entropy_coeff_decay=cfg.entropy_coeff_decay,
        )
    else:
        raise ValueError("Invalid game name")

    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()