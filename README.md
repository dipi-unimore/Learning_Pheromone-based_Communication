# Learning Pheromone-based Communication: Differentiation of Behaviour across Populations and Individuals 

This document details the code, parameters, and configurations used to obtain the results described in **XXX** article.

---

## ⚙️ Installation

1. Make sure you have Python >= 3.8 installed.

2. Create a virtual environment:

    ```bash
    python -m venv path_to_new_virtual_env 
    source path_to_new_virtual_env/bin/activate
    ```

3. Clone this repository:

    ```bash
    git clone https://github.com/dipi-unimore/Learning_Pheromone-based_Communication.git
    cd Learning_Pheromone-based_Communication
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

**Operating system**: The code was tested on Ubuntu 22.04, CPU only. 

---

## 📂 Project Structure

```plaintext
Learning_Pheromone-based_Communication/
├── agents                      # Algorithms folder
│   ├── IQLearning              # Indipendent Q-Learning implementation
│   │   └── config              # Algorithm configuration files
│   ├── NoLearning              # Deterministic policy implementation
│   └── utils                   # Utility functions
└── environments                # Multi-agent environments
    └── slime                   # Slime environment
        └── config              # Env configuration files 
```

---

## 🚀 Running the Code

### IQLearning
    
The main script is `slime_iql.py`, which accepts the following command-line arguments:

| Argument            | Type    | Default value | Description                                                           |
|---------------------|---------|---------------|-----------------------------------------------------------------------|
| `--train`           | bool    | False         | If `True`, training of the agents will be performed, else evaluation. |
| `--random_seed`     | int     | 42            | Change the default random seed for reproducibility.                   |
| `--random_seeds`    | int[]   | None          | Optional list of seeds; repeats each run once per seed.               |
| `--qtable_path`     | str     | None          | Path to a `.npy` file for loading the Q-table to perform evaluation.  |
| `--print_metrics`   | int     | 30            | Metrics printing frequency.                                           | 
| `--render`          | bool    | False         | If `True`, renders the environment visually.                          |
| `--experiments_dir` | str     | ""            | If provided, runs all `*-params-X.json` experiments in that directory in ascending `X`. |

**Example: Training run**

```bash
python slime_iql.py --train True --random_seed 99 
```

The **qtable** will be automatically put in the `./runs/weights` folder. 

**Example: Evaluation run**

```bash
python slime_iql.py --random_seed 99 --qtable_path ./runs/weights/file_name.npy --render True
```

**Example: Sequential experiments from `experiments/`**

If files like `env-params-1.json`, `env-params-2.json`, `learning-params-2.json` are present,
the script runs experiments `1`, then `2`, and for each missing config type it falls back to defaults.

```bash
python slime_iql.py --train True --experiments_dir experiments
```

**Example: Repeat with multiple seeds**

```bash
python slime_iql.py --train True --random_seeds 10 20 30
python slime_iql.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

### Deterministic Policy
    
The main script is `slime_deterministic.py`, which accepts the following command-line arguments:

| Argument            | Type   | Default value  | Description                                                              |
|---------------------|--------|----------------|--------------------------------------------------------------------------|
| `--random_seed`     | int    | 42             | Change the default random seed for reproducibility.                      |
| `--episodes`        | int    | 500            | Number of episodes.                                                      |
| `--render`          | bool   | False          | If `True`, renders the environment visually.                             |

**Run example**

```bash
python slime_deterministic.py --random_seed 99 --episodes 100 --render True 
```

---

## ⚙️ Key Parameters

### Environment

| Parameter                 | Values                         | Description                                  |
|---------------------------|-------------------------------------------|-----------------------------------|
| `World-size`                      | [(20x20), **(22x22)**, (25x25), (31x31)]  | Size of the grid world where agents move (torus).                                     |
| `Clutering-population` ($N_C$)    | [**20**, **14**, **10**, **6**, **0**]    | Number of clustering agents.                                                          |
| `Scattering-population` ($N_S$)   | [**0**, **6**, **10**, **14**, **20**]    | Number of scattering agents.                                                          |
| `Sniff-threshold`                 | [0.7, **0.9**, 1.1, 1.3]                  | Minimum amount of pheromone that can be <br> smelled by an agent.                     |
| `Sniff-patches`                   | [3, **5**, 8]                             | Number of 1-hop neighboring patches in which the agent <br> can smell the pheromone.  |
| `Wiggle-patches`                  | [3, **5**, 8]                             | Number of 1-hop neighboring patches the agent can move <br> randomly through.         |
| `Diffuse-area`                    | [**0.5**, 1.0, 1.5]                       | Standard deviation value of the Gaussian function used to <br> spread the pheromone in the environment.   |
| `Diffuse-radius`                  | 1.0                                       | Radius of the Gaussian function used to spread <br> the pheromone in the environment. |
| `Evaporation-rate`                | [0.8, 0.85, 0.9, **0.95**]                | Amount of pheromone not evaporating in the environment.                               |
| `Lay-area`                        | [0, **1**]                                | Number of patches in which the pheromone is released.                                 |
| `Lay-amount`                      | [1, 2, **3**, 5]                          | Amount of pheromone deposited evenly in `Lay-area`.                                   |

The values ​​in bold are the ones we used.

### Learning

| Parameter                 | Values                 | Description                   |
|---------------------------|------------------------|-------------------------------|
| `Cluster-threshold` ($c_{th}$)        | [**1**, 5, 10]                    | Minimum amount of agents within cluster-radius to check <br> clustering.  |
| `Cluster-radius`                      | [**1**, 2, 3]                     | Distance (in number of patches) centered in the agents to <br> control clustering, it is used for calculating rewards and <br> metrics.   |
| `Clustering-reward` ($r_C$)           | [0, 1, **10**, 100]               | Base reward given upon clustering.                                        |
| `Clustering-penalty` ($p_C$)          | [0, **-1**, -10, -100]            | Base penalty given for not clustering.                                    |
| `Scattering-reward` ($r_S$)           | [**0**, 1, 10, 100]               | Base reward given upon scattering.                                        |  
| `Scattering-penalty` ($p_S$)          | [0, **-1**, -10, -100]            | Base penalty given for not scattering.                                    |
| `Ticks-per-episode`                   | [250, **500**, 1000]              | Learning episode duration in simulation ticks.                            |
| `episodes`                            | [**3000**, 5000, 10000]           | Number of learning episodes.                                              |
| `learning-rate` ($\alpha$)            | [0.01, **0.025**, 0.05, 0.1]      | Magnitude of Q-values updates.                                            |
| `discount-factor` ($\gamma$)          | [**0.9**, 0.95, 0.99, 0.999]      | How much future rewards are given value.                                  |
| `epsilon-init` ($\epsilon_{init}$)    | 1.0                               | Initial exploration rate.                                                 |
| `epsilon-min`  ($\epsilon_{min}$)     | [5e−3, 1e−3, 5e−4, 1e−4, **0.0**] | Minimum value of epsilon.                                                 |
| `epsilon-decay` ($\lambda$)           | [**0.995**, 0.997, 0.999]         | How much epsilon lowers after each action, <br> it goes from ($\epsilon_{init}$) to ($\epsilon_{min}$).   |

The values ​​in bold are the ones we used.

---

## 🛠️ Configuration Files

The following .json configuration files are used to manage the experiment's parameters:

| File Name |	Purpose |
|-----------|---------|
|`/environments/slime/config/env-params.json` |	Defines the environment settings.|
|`/environments/slime/config/env_visualizer.json` |	Controls the rendering configuration for visualizing the environment|
|`/agent/IQLearning/config/learning-params.json` |	Contains learning-related parameters such as learning rate, epsilon decay, etc.|
|`/agent/IQLearning/config/logger-params.json` |	Configures the logging behavior and export mode.|

---

## 📈 Evaluation Metrics

All evaluation metrics described in the paper are automatically logged in `/runs/train` (for training) and in `/runs/eval`(for evaluation).

---

## 💾 Reproducibility

Ours paper results presents the average result of 10 identical experiments conducted on a population of 20 total agents. 

The random seeds we used: `[10, 20, 30, 40, 50, 60, 70 , 80, 90, 100]`.

---

## 📚 Citation

If you use this codebase in your research, please cite the following article:

> **XXX** 
> Authors: Davide Borghi, Stefano Mariani, and Franco Zambonelli  
> XXX 