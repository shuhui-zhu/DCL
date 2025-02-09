# Differentiable Commitment Learning
This repository implements DCL ([AISTATS 2025](https://github.com/shuhui-zhu/DCL)) in our paper [Learning to Negotiate via Voluntary  Commitment](https://github.com/shuhui-zhu/DCL).

## Set Up

* Clone this repository.

* Set up the virtual environment.

  ``` 
  conda create -n virtual_env python=3.10.9
  conda activate virtual_env
  pip install -r requirement.txt
  ```

## Navigation

The implementation of DCL is in `main` branch, decentralized DCL is in `decentralized_dcl` branch.

* `env/`: includes Prisoner's Dilemma, Repeated Purely Conflicting Game and Grid Game.

* `trainer`: includes DCL trainer of each game. 
* `utility`: includes replay buffer class, DCL agent class of each game.
* `main.py`: the runner file for implementing the algorithm with specified hyperparameters. 

## Running Experiments

```
python main.py --game "your_game_name" --with_constraints "with_or_without_IC"
```

### Examples

Follow the following examples for implementating DCL and DCL-IC. To reproduce results in  the paper [Learning to Negotiate via Voluntary  Commitment](https://github.com/shuhui-zhu/DCL), please use the hyperparameter set in supplementary materials (Appendix D). 

#### Train DCL on Prisoner's Dilemma

```
python main.py --game "IPD" --with_constraints "n" --batch_size 128
```

#### Train DCL-IC on Prisoner's Dilemma 

```
python main.py --game "IPD" --with_constraints "y" --batch_size 128
```

#### Train DCL on Repeated Purely Conflicting Game

```
python main.py --game "IPC" --with_constraints "n" --max_steps 16 --mega_step 2 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
```

#### Train DCL-IC on Repeated Purely Conflicting Game

```
python main.py --game "IPC" --with_constraints "y" --max_steps 16 --mega_step 2 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
```

#### Train DCL on Grid Game

```
python main.py --game "Grid" --with_constraints "n" --max_steps 16 --mega_step 1 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
```

#### Train DCL-IC on Grid Game

```
python main.py --game "Grid" --with_constraints "y" --max_steps 16 --mega_step 1 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
```

## Citation

If you want to cite this repository, please use the following citation:

```
# Add Later
```



## License

SPDX-License-Identifier: MIT
