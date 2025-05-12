# Differentiable Commitment Learning
This repository implements DCL ([AISTATS 2025](https://virtual.aistats.org/Conferences/2025)) in our paper [Learning to Negotiate via Voluntary  Commitment](https://arxiv.org/abs/2503.03866).

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

Follow the examples for implementating DCL and DCL-IC. To reproduce results in  the paper [Learning to Negotiate via Voluntary  Commitment](https://github.com/shuhui-zhu/DCL), please use the hyperparameter set in supplementary materials (Appendix D). 

* Train DCL on Prisoner's Dilemma

  ```
  python main.py --game "IPD" --with_constraints "n" --batch_size 128
  ```

* Train DCL-IC on Prisoner's Dilemma 

  ```
  python main.py --game "IPD" --with_constraints "y" --batch_size 128
  ```

* Train DCL on Repeated Purely Conflicting Game

  ```
  python main.py --game "IPC" --with_constraints "n" --max_steps 16 --mega_step 2 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
  ```

* Train DCL-IC on Repeated Purely Conflicting Game

  ```
  python main.py --game "IPC" --with_constraints "y" --max_steps 16 --mega_step 2 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
  ```

* Train DCL on Grid Game

  ```
  python main.py --game "Grid" --with_constraints "n" --max_steps 16 --mega_step 1 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
  ```

* Train DCL-IC on Grid Game

  ```
  python main.py --game "Grid" --with_constraints "y" --max_steps 16 --mega_step 1 --batch_size 512 --entropy_coeff 2.0 --temperature 1.0
  ```

## Citation

If you want to cite this repository, please use the following citation:

```
@InProceedings{pmlr-v258-zhu25b,
  title = 	 {Learning to Negotiate via Voluntary Commitment},
  author =       {Zhu, Shuhui and Wang, Baoxiang and Subramanian, Sriram Ganapathi and Poupart, Pascal},
  booktitle = 	 {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1459--1467},
  year = 	 {2025},
  editor = 	 {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume = 	 {258},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--05 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v258/main/assets/zhu25b/zhu25b.pdf},
  url = 	 {https://proceedings.mlr.press/v258/zhu25b.html},
  abstract = 	 {The partial alignment and conflict of autonomous agents lead to mixed-motive scenarios in many real-world applications. However, agents may fail to cooperate in practice even when cooperation yields a better outcome. One well known reason for this failure comes from non-credible commitments. To facilitate commitments among agents for better cooperation, we define Markov Commitment Games (MCGs), a variant of commitment games, where agents can voluntarily commit to their proposed future plans. Based on MCGs, we propose a learnable commitment protocol via policy gradients. We further propose incentive-compatible learning to accelerate convergence to equilibria with better social welfare. Experimental results in challenging mixed-motive tasks demonstrate faster empirical convergence and higher returns for our method compared with its counterparts. Our code is available at \url{https://github.com/shuhui-zhu/DCL.}}
}
```

## License
See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT.