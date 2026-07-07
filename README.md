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
│   ├── IQLearning              # Indipendent Q-Learning implementation
│   ├── CoQLearning             # Collaborative Q-Learning implementation
│   │   └── config              # Algorithm configuration files
│   ├── QMIXLearning            # QMIX (Value Decomposition) implementation
│   │   └── config              # Algorithm configuration files
│   ├── NoLearning              # Deterministic policy implementation
│   └── utils                   # Utility functions
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

If `--random_seeds` is provided, it is used as the source of seeds (even if it has a single element).
If `--random_seeds` is not provided, the script falls back to `--random_seed` for backward compatibility.

### CoQLearning

The collaborative variant is exposed through `slime_coql.py` and keeps the same CLI surface as `slime_iql.py`.
Its default learning configuration lives in `agents/CoQLearning/config/learning-params.json`.

```bash
python slime_coql.py --train True --random_seeds 10 20 30
python slime_coql.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

`agents/CoQLearning/config/learning-params.json` adds a `collaboration` block:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enables collaborative updates during training. |
| `share_every_steps` | `1` | Share information every N simulation ticks. |
| `recipient_selector` | `"all"` | Share with `all`, `nearby`, or `similar` agents. |
| `nearby_radius` | `1` | Toroidal square radius used when `recipient_selector="nearby"`. |
| `share_rate` | `0.1` | Blend factor used by collaborative updates. |
| `reward_weight` | `1.0` | Multiplier applied to shared rewards before shaping the update. |
| `shared_information` | all `true` | Independently enables `observation`, `action`, `reward`, and `q_values` sharing. |

To avoid exploding the tabular state space, CoQL does not append peer information to the observation.
Instead, it uses compact summaries during the Q-update. The following subsections detail each integration strategy.

#### A. Shared Observations

When `shared_information.observation=true`, the recipient agent integrates peers' observations without enlarging the state space:

1. Identify the observation state indices for each peer: $s_{\text{peer}_i} = \text{env.convert\_observation}(o_{\text{peer}_i})$
2. **Look up the RECIPIENT's own Q-value row for each peer's observed state**: $Q_{\text{recipient}}(s_{\text{peer}_i}, \cdot)$  
   (This is key: use the recipient's learned knowledge about what those states mean to *itself*.)
3. Average the Q-rows across all peers: $\bar{Q}_{\text{shared}} = \frac{1}{|\text{peers}|} \sum_i Q_{\text{recipient}}(s_{\text{peer}_i}, \cdot)$
4. Blend the shared Q-row with the recipient's current Q-row using the configured `share_rate`:
   $$Q_{\text{recipient}}(s_{\text{recipient}}, \cdot) \leftarrow (1 - r_s) \cdot Q_{\text{recipient}}(s_{\text{recipient}}, \cdot) + r_s \cdot \bar{Q}_{\text{shared}}$$
   where $r_s$ is the `share_rate`.

**Intuition**: "What would I think about the states my peers are observing?" Uses only the recipient's own learned beliefs about those states, without being influenced by how peers value them. This is a more conservative form of generalization.

#### B. Shared Actions

When `shared_information.action=true`, the recipient integrates peers' behavioral history via majority voting:

1. Collect all peers' previous actions (one per peer, if available) and current actions: $\mathcal{A}_{\text{shared}} = \{a_{\text{prev}}, a_{\text{curr}}\}$ per peer
2. Compute the most frequent action in the pool: $a_{\text{majority}} = \text{argmax}_a |\{a' \in \mathcal{A}_{\text{shared}} : a' = a\}|$
3. Increment the Q-value for that action in the recipient's current state by the `share_rate`:
   $$Q_{\text{recipient}}(s_{\text{recipient}}, a_{\text{majority}}) \leftarrow Q_{\text{recipient}}(s_{\text{recipient}}, a_{\text{majority}}) + r_s$$

**Intuition**: A small bonus nudges the recipient's policy toward actions that peers find useful, acting as a lightweight coordination pressure without distorting the learned value landscape.

#### C. Shared Rewards

When `shared_information.reward=true`, the recipient shapes its Q-update using peers' reward signals:

1. Compute the average peer reward: $\bar{r}_{\text{shared}} = \frac{1}{|\text{peers}|} \sum_i r_i$
2. Scale the shared reward by the configured `reward_weight`: $r_{\text{shaped}} = w_r \cdot \bar{r}_{\text{shared}}$ (default $w_r = 1.0$)
3. Form a shaped TD target using the shared reward:
   $$\bar{Q}(s_{\text{recipient}}) = (1 - \alpha) \cdot Q(s_{\text{recipient}}, a_{\text{recipient}}) + \alpha \cdot (r_{\text{shaped}} + \gamma \cdot \max_a Q(s_{\text{recipient}}, a))$$
4. Blend the shaped target with the recipient's current Q-value:
   $$Q_{\text{recipient}}(s_{\text{recipient}}, a_{\text{recipient}}) \leftarrow (1 - r_s) \cdot Q(s_{\text{recipient}}, a_{\text{recipient}}) + r_s \cdot \bar{Q}(s_{\text{recipient}})$$

**Intuition**: Peers' successes and failures (encoded in their rewards) can guide the recipient's exploration by offering alternative value signals. The `reward_weight` lets you scale how much external rewards influence your own learning.

#### D. Shared Q-Values

When `shared_information.q_values=true`, the recipient directly fuses peers' learned knowledge:

1. For each peer, identify its current observation state: $s_{\text{peer}_i}$
2. **Retrieve each PEER's Q-value row for that peer's state**: $Q_{\text{peer}_i}(s_{\text{peer}_i}, \cdot)$  
   (This is key: use each peer's own learned estimates, not the recipient's.)
3. Average the peers' Q-rows:
   $$\bar{Q}_{\text{consensus}} = \frac{1}{|\text{peers}|} \sum_i Q_{\text{peer}_i}(s_{\text{peer}_i}, \cdot)$$
4. Blend the consensus row directly into the recipient's current-state Q-values:
   $$Q_{\text{recipient}}(s_{\text{recipient}}, \cdot) \leftarrow (1 - r_s) \cdot Q_{\text{recipient}}(s_{\text{recipient}}, \cdot) + r_s \cdot \bar{Q}_{\text{consensus}}$$

**Intuition**: This is the strongest form of collaboration, acting as a distributed consensus mechanism. Peers vote on action values using **their own learned estimates**, allowing the recipient to benefit from the exploration and learning progress of others. This is more aggressive than A because it directly adopts peers' value judgments.

---

**Comparison: Why A and D are Different**

Sections A and D use structurally identical blending but consult **different Q-tables**:

| Aspect | A (Shared Observations) | D (Shared Q-Values) |
|--------|----------------------|-------------------|
| **What is averaged?** | Recipient's Q-values at peer-observed states | Peers' own Q-values at their own states |
| **Formula** | $\frac{1}{n}\sum_i Q_{\text{recipient}}(s_{\text{peer}_i}, \cdot)$ | $\frac{1}{n}\sum_i Q_{\text{peer}_i}(s_{\text{peer}_i}, \cdot)$ |
| **Information source** | Recipient's internalized model | Peers' learned experiences |
| **Risk** | Self-reinforcing if recipient has incorrect beliefs | May adopt peers' bad habits if they're coherently wrong |
| **Best used when** | You trust your own learning and want to generalize to peer observations | You want to explicitly transfer knowledge from peers |

**Example**: Suppose a recipient agent and peer observe the **same state**  ($s_{\text{recipient}} = s_{\text{peer}}$), but are at **different positions**:
- **A**: The recipient asks "What does *I* think about this state?" and uses my own Q-values.
- **D**: The recipient asks "What does *my peer* think about this state?" and uses the peer's Q-values.

If the peer has explored more and learned better, **D** will bootstrap the recipient toward the peer's policy. If the recipient wants to stay true to its own learning, **A** is more conservative.

---

**General notes on integration:**
- All integrations use a **per-step Q-table snapshot** to avoid order-dependent artifacts. Each collaborative step reads from the Q-table as it existed at tick $t$, not including updates made earlier in the same tick.
- The `share_every_steps` parameter controls the frequency: sharing happens when `tick % share_every_steps == 0`.
- The `recipient_selector` determines which peers contribute (see sections below).
- Regardless of `recipient_selector`, information is shared **only among agents of the same kind**: cluster agents (`env.learners[*]["mode"] == "c"`) share only with cluster agents, and scatter agents (`env.learners[*]["mode"] == "s"`) share only with scatter agents.
- All sharing is **disabled during evaluation** (test phase).

### QMIXLearning

QMIX (Value Decomposition Networks for Cooperative Multi-Agent Reinforcement Learning) decomposes the global Q-function into individual agent Q-functions and a learned mixing network.

The main script is `slime_qmix.py` and accepts the same CLI arguments as `slime_coql.py`:

```bash
python slime_qmix.py --train True --random_seed 99
python slime_qmix.py --train True --random_seeds 10 20 30
python slime_qmix.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

**QMIX-specific configuration** in `agents/QMIXLearning/config/learning-params.json`:

| Field | Default | Description |
|-------|---------|-------------|
| `alpha` | `0.025` | Learning rate for Q-table updates |
| `gamma` | `0.9` | Discount factor |
| `epsilon` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.1` | Minimum exploration rate |
| `decay_type` | `"log"` | Type of epsilon decay (`"log"` or `"linear"`) |
| `decay` | `0.9987` | Decay rate parameter |
| `train_episodes` | `3000` | Number of training episodes |
| `test_episodes` | `100` | Number of evaluation episodes |
| `mixing_learning_rate` | `0.025` | Learning rate for the mixing network |

**Key differences from IQL/CoQL:**
- Individual agents maintain their own Q-tables (like IQL) for decentralized execution
- A learned mixing network aggregates individual Q-values during training: `Q_total = W * [Q_1, Q_2, ..., Q_n] + V`
- The mixing network learns linear weights that favor good cooperative actions
- Evaluation uses only the individual Q-tables (decentralized policy)

**Algorithm outline:**
1. Each agent selects actions using its own Q-table (ε-greedy)
2. Individual Q-values are updated using standard Q-learning
3. Mixing network combines individual Q-values into a global estimate
4. Mixing network weights are updated using the global TD error
5. This encourages agents to learn policies that work well when mixed together

---

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
|`/agent/CoQLearning/config/learning-params.json` |	Contains collaborative learning settings such as what, when, and with whom agents share information.|
|`/agent/CoQLearning/config/logger-params.json` |	Configures the logging behavior for CoQL runs.|
|`/agents/QMIXLearning/config/learning-params.json` |	Contains Q-learning parameters and mixing network learning rate for QMIX.|
|`/agents/QMIXLearning/config/logger-params.json` |	Provides run metadata naming defaults for QMIX experiments.|

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