![relex logo](relex_logo.svg)
==============================
**RELEX** - Reinforcement Learning Experiments

# Intro

RELEX project was created with three ideas in mind:
To teach myself Reinforcement Learning by implementing algorithms from scratch;
To teach others who struggle to understand some detailed aspects of reinforcement learning algorithms;
To create a space where I can experiment with innovative ideas and environments for research purposes.

Therefore, this is not a production-ready or deployment-ready library. But if you are looking for some (hopefully) easy-to-understand implementations of RL from scratch or some inspirations for study/research/paper - probably this is the right place :)

# The idea behind RELEX

I wanted to keep implementations in this library as simple as possible, even if this means that algorithms will work slowly (because of lack of parallelism). I wanted to have something easy-to-debug, break and play with instead of a lightning-fast but hard-to-follow tool.
For example, PPO implementation, or AC in theory, is embarrassingly parallel (multiple independent agents, gathering trajectories using copies of the environment), but RELEX version is single-threaded, so you can easily follow what's going on.

# Sources for algorithms

## Papers

1. Value-based algorithms:
   1. DQN: [Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. _nature_, _518_(7540), 529-533.](https://daiwk.github.io/assets/dqn.pdf)
2. Policy gradient:
   1. PPO
      1. [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.](https://arxiv.org/pdf/1707.06347.pdf)
      2. [Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.](https://arxiv.org/abs/1506.02438)
   2. AC
      1. [Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937). PMLR.](https://arxiv.org/pdf/1602.01783.pdf)
      2. [Konda, V., & Tsitsiklis, J. (1999). Actor-critic algorithms. Advances in neural information processing systems, 12.](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
   3. VPG
      1. [Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
      2. [Abbeel, P. (2016). OPTIMIZING EXPECTATIONS: FROM DEEP REINFORCEMENT LEARNING TO STOCHASTIC COMPUTATION GRAPHS (Doctoral dissertation, University of California, Berkeley).](http://joschu.net/docs/thesis.pdf)
   4. SAC
      1. [Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, July). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning (pp. 1861-1870). PMLR.](https://arxiv.org/abs/1801.01290)
      2. [Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.](https://arxiv.org/abs/1812.05905)

## Tutorials/other libraries/books

1. [Spinning up AI](https://spinningup.openai.com/en/latest/index.html)
2. Morales, M. (2020). Grokking deep reinforcement learning. Manning Publications.
3. Winder, P. (2021). Reinforcement learning: Industrial applications of intelligent agents.

# Algorithms

## TF2+ implementations

Policy gradient algorithms:
- [x] PPO
- [x] AC
- [x] VPG

Hybrid algorithms:
- [x] SAC

Value-based algorithms:
- [x] DQN (vanilla)
- [ ] DDQN
- [ ] Dueling DQN

## Pytorch implementations

TODO - top priority after DQN/DDQN/Dueling DQN

# Project structure

This project is structured according to the Data Science Cookiecutter schema. Below you can find the description of the main directories from RL / ML perspective:

1. **experiments/** - contains self-contained scripts under MLFLow monitoring, each with a full experimentation pipeline.
   1. Experiments are divided into:
      1. **policy gradient** - generic experiments to check/debug/learn how algorithms work; 
      2. **stock_trading**- experiments connected with stock markets; 
      3. **Various** - uncategorized experiments utilizing e.g. external libraries like Stable-Baselines 3.
   2. In each experiment the following operations are performed:
         1. **Check** the algorithm performance before training.
         2. **Evaluate "benchmark agents"** (e.g., always predict mean, random choice). 
         3. **Train** the main agent 
         4. **Evaluate** agent after training. 
         5. **Make statistical comparison** - nonparametric Kruskal test followed by post-hoc tests between pairs of agents (benchmarks vs main).
2. **notebooks/** - personally, I'm against using notebooks to store finalized code. However, sometimes - for educational purposes or when preparing a scientific paper, a "notebook" (quotation intentional - notebook in a strict sense) form comes as a natural choice. Most notebooks in this folder are part of research papers (in writing, in review, or already published) or are "scratchpads" that will later be rewritten as self-contained scripts.
3. **src/** dir that is divided into:
   1. **algorithms** - contains implementations of various RL algorithms, divided into categories (policy gradient and others).
   2. **models** - contains basic, shared neural network architectures used by various models. E.g.: policy/value networks used by all policy gradient algorithms.
   3. **experiments** - contains various utilities for experiments that help automate the experimentation process.
   4. **envs** - various experimental implementations of environments. For example: "project allocation environment" used in one of the papers, or "stock trading" envs.

# Where to start?

1. If you are interested in a particular algorithm implementation - go to src/algorithms.
2. If you are interested in shared/common neural networks (like policy or value nets) - go to src/models
3. If you want to look at experiments:
   1. **experiments/** - contains self-containing scripts for each algorithm or problem. 
   2. **notebooks/** - also include experiments, sometimes half-done, not working, or in progress, in the form of notebooks.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Publications

Below you can find publications utilizing Relex library.

Below you can find publications utilizing Relex. The list will be gradually updated as I develop both - the library and write new papers (which can take a looooot of time, sometimes a year, before a review). 

I will be glad, if you could cite the papers in your own work if you use Relex somewhere.


| **Paper** | **Status**                  |
|-----------|-----------------------------|
|Wójcik, F. (2022). *Utilization of deep reinforcement learning for discrete resource allocation problem in project management - a simulation experiment*. Informatyka Ekonomiczna=Business Informatics. | In review (as of July 2022) |
