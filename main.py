
from env.repeated_game import IteratedPrisonersDilemma, IteratedPureConflict
from env.grid_game import GridSocialDilemmaEnv

from utility.agents.agent_class_repeated_games import DCL_Agent_Repeated_Games
from utility.agents.agent_class_grid_game import DCL_Agent_Grid_Game

from trainer.dcl_repeated_games import Repeated_Game_Trainer
from trainer.dcl_grid_game import Grid_Game_Trainer

import wandb
import argparse
import os

os.environ['WANDB_MODE'] = 'disabled'

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--array_task_id', type=int, default=0)
parser.add_argument('--with_constraints', type=str, default='n')
parser.add_argument('--game', type=str, default='IPD')
parser.add_argument("--max_steps", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--N_iterations", default=10000, type=int) # Number of iterations to train the model
parser.add_argument("--hidden_dim", default=8, type=int) # Hidden dimension of the neural network
parser.add_argument("--lr_critic", default=7e-4, type=float) # Learning rate of the Critic
parser.add_argument("--lr_actor", default=4e-4, type=float) # Learning rate of the Actor
parser.add_argument("--temperature", default=10.0, type=float) # Temperature for the softmax
parser.add_argument("--num_iter_per_batch", default=1, type=int) # Number of updates per iteration
parser.add_argument("--is_entropy", default='y', type=str) # whether to use entropy regularization
parser.add_argument("--entropy_coeff", default=1.0, type=float) # Entropy regularization coefficient
parser.add_argument("--entropy_coeff_decay", default=0.0005, type=float) # Entropy regularization coefficient decay
parser.add_argument("--temperature_decay", default=0.05, type=float) # Temperature decay
parser.add_argument("--mega_step", default=1, type=int)
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument("--perturb", default=1e-3, type=float) # perturbation for probs (avoid numerical issues)

if __name__ == '__main__':
    const_flag = parser.parse_args().with_constraints
    with_constraints = True if const_flag == 'y' else False
    gamma = parser.parse_args().gamma
    max_steps = parser.parse_args().max_steps
    array_task_id = parser.parse_args().array_task_id
    id = wandb.util.generate_id()
    game_name = parser.parse_args().game
    batch_size = parser.parse_args().batch_size
    N_iterations = parser.parse_args().N_iterations
    total_steps = N_iterations*batch_size
    N_episodes = int(total_steps/max_steps)
    hidden_dim = parser.parse_args().hidden_dim
    lr_critic = parser.parse_args().lr_critic
    lr_actor = parser.parse_args().lr_actor
    temperature = parser.parse_args().temperature
    is_entropy = True if parser.parse_args().is_entropy == 'y' else False
    entropy_coeff = parser.parse_args().entropy_coeff
    entropy_coeff_decay = parser.parse_args().entropy_coeff_decay
    temperature_decay = parser.parse_args().temperature_decay
    mega_step = parser.parse_args().mega_step
    num_iter_per_batch = parser.parse_args().num_iter_per_batch
    grid_size = parser.parse_args().grid_size
    
    assert batch_size % max_steps == 0  # batch_size should be divisible by max_steps

    wandb.init(project="DCL",group=game_name)

    if game_name == 'IPD':
        environment = IteratedPrisonersDilemma
    elif game_name == 'IPC':
        environment = IteratedPureConflict
    elif game_name == 'Grid':
        environment = GridSocialDilemmaEnv
    else:
        raise ValueError("Invalid game name")
    
    if game_name == 'IPD' or game_name == 'IPC':
        trainer = Repeated_Game_Trainer(env=environment, agent_class=DCL_Agent_Repeated_Games, mega_step=mega_step,
                                        gamma=gamma, max_steps=max_steps,temperature=temperature, temperature_decay=temperature_decay,\
                                        hidden_dim=hidden_dim, lr_critic=lr_critic, lr_actor=lr_actor,\
                                        with_constraints = with_constraints, buffer_length=batch_size, num_iter_per_batch=num_iter_per_batch,\
                                        batch_size=batch_size, N_agents=2, N_episodes=N_episodes,\
                                        is_entropy=is_entropy, entropy_coeff=entropy_coeff, entropy_coeff_decay=entropy_coeff_decay)
    elif game_name == 'Grid':
        trainer = Grid_Game_Trainer(env=environment, agent_class=DCL_Agent_Grid_Game, gamma=gamma,\
                                    max_steps=max_steps,temperature=temperature, temperature_decay=temperature_decay,\
                                    hidden_dim=hidden_dim, lr_critic=lr_critic, lr_actor=lr_actor,\
                                    with_constraints = with_constraints, buffer_length=batch_size, num_iter_per_batch=num_iter_per_batch,\
                                    batch_size=batch_size, N_agents=2, N_episodes=N_episodes, grid_size=grid_size,\
                                    is_entropy=is_entropy, entropy_coeff=entropy_coeff, entropy_coeff_decay=entropy_coeff_decay)
    else:
        raise ValueError("Invalid game name")
    
    trainer.train()
    wandb.finish()