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
│   ├── QMIXLearningNN          # Neural-network QMIX implementation
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

`QMIXLearning` implements a **tabular, QMIX-inspired cooperative learner** for the Slime environment.

The implementation follows the centralized-training/decentralized-execution idea:

- during execution, every learner selects actions from its own tabular Q-function;
- during training, learners of the same kind can receive a shared TD signal computed by a same-kind mixing network;
- cluster learners and scatter learners are never mixed together for learning updates.

This implementation is different from `QMIXLearningNN`. In the neural-network version, PyTorch backpropagation sends the mixed TD loss through the mixer into the agent networks automatically. In the tabular version, the individual Q-functions are NumPy Q-tables, so there is no autograd path from the mixer back into the individual Q-values. For that reason, tabular QMIX explicitly defines how the same-mode mixed TD error is assigned back into the agents' Q-table entries.

The main script is `slime_qmix.py` and accepts the same CLI arguments as `slime_coql.py`:

```bash
python slime_qmix.py --train True --random_seed 99
python slime_qmix.py --train True --random_seeds 10 20 30
python slime_qmix.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

#### Tabular mixer

For each learner mode, the mixer is a simple linear tabular mixer:

```text
Q_total = W_a dot [Q_1, Q_2, ..., Q_n] + V_a
```

where:

- `n` is the number of agents in the same-mode group;
- `Q_i` is the selected Q-table value for same-mode agent `i`;
- `W_a` is the mixer weight vector associated with action `a`;
- `V_a` is an action-specific value baseline.

The mixer is not used directly for evaluation-time action selection. During evaluation, each agent acts greedily from its own Q-table. However, when mixed-TD credit assignment is enabled, the mixer affects the Q-tables during training, and therefore affects the final decentralized policies indirectly.

#### Training outline

At each training step:

1. Each learner observes its local state.
2. If a previous state/action exists, the learner may receive a local tabular TD update.
3. The learner selects an action using epsilon-greedy action selection from its own Q-table.
4. After all learners have acted, the implementation groups learners by mode.
5. For each mode independently:
   - collect same-mode previous Q-values;
   - collect same-mode previous actions;
   - collect same-mode next-state max Q-values;
   - sum same-mode rewards;
   - compute a same-mode mixed TD target;
   - update the same-mode mixer, if configured;
   - optionally feed the same-mode mixed TD error back into the same-mode agents' Q-table entries.

The local TD update is:

```text
local_td_error = reward_i + gamma * max_a Q_i(next_state_i, a) - Q_i(prev_state_i, prev_action_i)

Q_i(prev_state_i, prev_action_i) +=
    alpha * local_td_weight * local_td_error
```

The mixed TD update is:

```text
mode_td_error = same_mode_mixed_target - same_mode_mixed_q

Q_i(prev_state_i, prev_action_i) +=
    alpha * mixed_td_weight * credit_share_i * mode_td_error
```

The value of `credit_share_i` depends on the configured mixed-TD strategy.

#### Configuration file

Tabular QMIX uses:

```text
agents/QMIXLearning/config/learning-params.json
```

Recommended default configuration:

```json
{
    "alpha": 0.025,
    "gamma": 0.9,
    "epsilon": 1.0,
    "epsilon_min": 0.1,
    "decay_type": "log",
    "decay": 0.9987,
    "train_episodes": 3000,
    "test_episodes": 100,
    "mixing_learning_rate": 0.025,
    "qmix_credit_assignment": {
        "enabled": true,
        "qtable_update_mode": "hybrid",
        "mixed_td_strategy": "equal_share",
        "local_td_weight": 1.0,
        "mixed_td_weight": 1.0,
        "update_mixer": true,
        "weight_epsilon": 1e-12
    }
}
```

Use `mixing_learning_rate` for tabular QMIX. Do **not** rename it to `mixer_learning_rate`; `mixer_learning_rate` is used by the neural-network QMIX implementation.

The `qmix_credit_assignment` block is specific to tabular QMIX. Do **not** add it to the neural-network QMIX configuration.

#### Base learning parameters

| Field | Default | Description |
|---|---:|---|
| `alpha` | `0.025` | Base Q-table learning rate. It scales both local TD updates and mixed-TD credit-assignment updates. |
| `gamma` | `0.9` | Discount factor used in local and mixed TD targets. |
| `epsilon` | `1.0` | Initial exploration rate for epsilon-greedy action selection. |
| `epsilon_min` | `0.1` | Minimum exploration rate. |
| `decay_type` | `"log"` | Type of epsilon decay. Supported values are `"log"` and `"linear"`. |
| `decay` | `0.9987` | Epsilon decay parameter. |
| `train_episodes` | `3000` | Number of training episodes. |
| `test_episodes` | `100` | Number of evaluation episodes. |
| `mixing_learning_rate` | `0.025` | Learning rate for the same-mode tabular mixers. This is meaningful only when `qmix_credit_assignment.update_mixer` is `true`. |

#### `qmix_credit_assignment` parameters

| Field | Default | Description |
|---|---:|---|
| `enabled` | `true` | Master switch for mixed-TD Q-table feedback. If set to `false`, the code forces `qtable_update_mode` to `"local_only"`. This does **not** automatically disable mixer training; set `update_mixer` to `false` as well if you want an IQL-like run with no useful mixer. |
| `qtable_update_mode` | `"hybrid"` | Selects how Q-table entries are updated. Supported values are `"local_only"`, `"equal_share"`, `"mixer_weight_share"`, `"full_shared_td"`, and `"hybrid"`. |
| `mixed_td_strategy` | `"equal_share"` | Selects the mixed-TD credit strategy used by `"hybrid"`. Supported values are `"equal_share"`, `"mixer_weight_share"`, and `"full_shared_td"`. Ignored by non-hybrid modes. |
| `local_td_weight` | `1.0` | Multiplier for the local independent-Q-learning TD update. Meaningful only for `"local_only"` and `"hybrid"`. |
| `mixed_td_weight` | `1.0` | Multiplier for the mixed-TD Q-table feedback update. Meaningful for `"equal_share"`, `"mixer_weight_share"`, `"full_shared_td"`, and `"hybrid"`. |
| `update_mixer` | `true` | Whether to train the same-mode mixer weights using the same-mode mixed TD error. Usually should be `true` whenever mixed TD feedback is used. |
| `weight_epsilon` | `1e-12` | Numerical threshold used by `"mixer_weight_share"`. If the relevant mixer weights are too close to zero, the implementation falls back to equal sharing. Ignored by other strategies. |

#### Q-table update modes

##### `local_only`

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "local_only",
    "local_td_weight": 1.0,
    "update_mixer": true
}
```

Each agent updates its own Q-table using only its own local TD error:

```text
Q_i += alpha * local_td_weight * local_td_error
```

The mixer may still be trained if `update_mixer` is `true`, but the mixer TD error is not written back into the Q-tables. Therefore, in this mode the mixer is auxiliary and does not affect the decentralized policies.

Use this mode as an independent-Q-learning-style baseline or ablation inside the tabular QMIX code path.

To remove the practical effect of the mixer, use:

```json
"qmix_credit_assignment": {
    "enabled": false,
    "qtable_update_mode": "local_only",
    "local_td_weight": 1.0,
    "update_mixer": false
}
```

##### `equal_share`

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "equal_share",
    "mixed_td_weight": 1.0,
    "update_mixer": true
}
```

This mode skips the local TD update and updates each same-mode agent's Q-table with an equal fraction of the same-mode mixed TD error:

```text
credit_share_i = 1 / same_mode_group_size

Q_i += alpha * mixed_td_weight * credit_share_i * mode_td_error
```

This is the simplest cooperative credit-assignment rule. It is most appropriate when same-mode agents are homogeneous and symmetric, or when there is no reliable reason to assign more credit to one same-mode agent than another.

It may be too crude when agents have different roles, observations, or contributions, because every same-mode agent receives the same share of the group error.

##### `mixer_weight_share`

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "mixer_weight_share",
    "mixed_td_weight": 1.0,
    "update_mixer": true,
    "weight_epsilon": 1e-12
}
```

This mode skips the local TD update and distributes the same-mode mixed TD error according to the current same-mode mixer weights:

```text
credit_share_i proportional to abs(mixer_weight_i)
```

The absolute value is used so that negative weights in the simple linear mixer do not invert the sign of the TD correction. Shares are normalized to sum to one. If the relevant mixer weights are all near zero, the implementation falls back to equal sharing.

This mode is meaningful when the mixer weights have become informative and can be interpreted as a rough estimate of each same-mode agent's contribution to the mixed value.

It can be fragile early in training because the mixer weights are initially random and may not yet represent useful credit assignment. For that reason, this strategy is often safer inside `"hybrid"` than as a pure update mode.

##### `full_shared_td`

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "full_shared_td",
    "mixed_td_weight": 0.1,
    "update_mixer": true
}
```

This mode skips the local TD update and applies the full same-mode mixed TD error to every same-mode agent:

```text
credit_share_i = 1

Q_i += alpha * mixed_td_weight * mode_td_error
```

This is the strongest cooperative update. Every same-mode agent receives the full group-level correction.

It can be useful when you want to aggressively align same-mode agents around the same collective signal. However, it can easily over-amplify the update, because the total Q-table change scales with the number of same-mode agents. For that reason, `mixed_td_weight` should usually be smaller for this mode than for `"equal_share"` or `"mixer_weight_share"`.

##### `hybrid`

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "hybrid",
    "mixed_td_strategy": "equal_share",
    "local_td_weight": 1.0,
    "mixed_td_weight": 1.0,
    "update_mixer": true
}
```

This is the recommended default.

In hybrid mode, each same-mode agent receives both:

1. a local independent-Q-learning TD update;
2. a same-mode mixed TD update.

```text
Q_i += alpha * local_td_weight * local_td_error
Q_i += alpha * mixed_td_weight * credit_share_i * mode_td_error
```

The `mixed_td_strategy` field chooses how the mixed TD error is shared:

```json
"mixed_td_strategy": "equal_share"
```

or:

```json
"mixed_td_strategy": "mixer_weight_share"
```

or:

```json
"mixed_td_strategy": "full_shared_td"
```

Hybrid mode is usually the safest choice because the local reward signal remains available while the same-mode mixer adds a cooperative training signal.

#### Meaningful configuration combinations

##### Recommended default: local learning plus equal cooperative signal

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "hybrid",
    "mixed_td_strategy": "equal_share",
    "local_td_weight": 1.0,
    "mixed_td_weight": 1.0,
    "update_mixer": true
}
```

Use this as the default starting point. Agents learn from their own rewards and also receive a normalized same-mode cooperative TD signal.

##### Conservative hybrid: mostly local, weak cooperative correction

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "hybrid",
    "mixed_td_strategy": "equal_share",
    "local_td_weight": 1.0,
    "mixed_td_weight": 0.1,
    "update_mixer": true
}
```

Use this when mixed TD feedback is noisy or destabilizes training. The local Q-learning signal dominates, while the mixer provides a weaker same-mode correction.

##### Weight-based hybrid

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "hybrid",
    "mixed_td_strategy": "mixer_weight_share",
    "local_td_weight": 1.0,
    "mixed_td_weight": 1.0,
    "update_mixer": true,
    "weight_epsilon": 1e-12
}
```

This keeps the local TD update and lets the mixer decide how much of the mixed TD error each same-mode agent receives.

This is more adaptive than equal sharing, but it relies on the mixer weights being meaningful. If training is unstable, prefer `"equal_share"` first.

##### Strong cooperative hybrid

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "hybrid",
    "mixed_td_strategy": "full_shared_td",
    "local_td_weight": 1.0,
    "mixed_td_weight": 0.1,
    "update_mixer": true
}
```

Every same-mode agent receives the full same-mode mixed TD error in addition to the local TD update. This is a strong coupling configuration and should usually use a smaller `mixed_td_weight`.

##### Pure cooperative equal-share update

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "equal_share",
    "mixed_td_weight": 1.0,
    "update_mixer": true
}
```

This removes the local TD update and trains Q-tables only through the same-mode mixed TD error with equal sharing.

Use this to test whether same-mode cooperative learning alone is sufficient.

##### Pure cooperative mixer-weight update

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "mixer_weight_share",
    "mixed_td_weight": 1.0,
    "update_mixer": true,
    "weight_epsilon": 1e-12
}
```

This trains Q-tables only through the same-mode mixed TD error, distributed according to mixer weights.

This is meaningful as an experiment, but it is more fragile than hybrid mode because early mixer weights may not provide reliable credit assignments.

##### Pure cooperative full-shared update

```json
"qmix_credit_assignment": {
    "enabled": true,
    "qtable_update_mode": "full_shared_td",
    "mixed_td_weight": 0.1,
    "update_mixer": true
}
```

This trains Q-tables only through the full same-mode mixed TD error.

Because every same-mode agent receives the full error, this can produce large updates. Prefer a smaller `mixed_td_weight`.

#### Difference from CoQL

CoQL shares information explicitly between same-mode peers. For example, it may blend observations, actions, rewards, or Q-values from selected same-mode agents into the recipient's Q-table.

Tabular QMIX shares information differently. It does not copy peer Q-values directly. Instead, it computes a same-mode group TD error through a same-mode mixer and then assigns that group-level TD error back into the same-mode agents' Q-table entries.

```text
CoQL:
    same-mode peer data
        -> direct Q-table collaboration

Tabular QMIX:
    same-mode rewards + same-mode Q-values
        -> same-mode mixed TD error
        -> configurable Q-table credit assignment
```

#### Difference from QMIXLearningNN

The `qmix_credit_assignment` block is only for tabular QMIX.

Neural-network QMIX does not use this block because PyTorch backpropagation already propagates the mixed TD loss through the mixer into the agent networks.

Use:

```text
agents/QMIXLearning/config/learning-params.json
    -> use mixing_learning_rate
    -> use qmix_credit_assignment

agents/QMIXLearningNN/config/learning-params.json
    -> use mixer_learning_rate
    -> do not use qmix_credit_assignment
```

---

### QMIXLearningNN

`slime_qmix_nn.py` is a neural-network variant of QMIX implemented with **PyTorch**.

It keeps the same CLI and experiment batching style as the other runners, but uses:
- per-agent Q-networks (MLPs) and a QMIX mixer network
- **Alternative C** training: replay buffer + mini-batch updates + target networks
- **Double-Q style** target computation (online selects actions, target evaluates)
- `device` auto-selection for CUDA/MPS/CPU
- flat `.npy` serialization via `Logger` (Option A compatibility path)

```bash
python slime_qmix_nn.py --train True --random_seed 99
python slime_qmix_nn.py --train True --random_seeds 10 20 30
python slime_qmix_nn.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

**QMIX-NN configuration** in `agents/QMIXLearningNN/config/learning-params.json`:

| Field | Default | Description |
|-------|---------|-------------|
| `alpha` | `0.001` | Learning rate for the per-agent neural networks |
| `gamma` | `0.9` | Discount factor |
| `epsilon` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.1` | Minimum exploration rate |
| `decay_type` | `"log"` | Type of epsilon decay (`"log"` or `"linear"`) |
| `decay` | `0.9987` | Decay rate parameter |
| `train_episodes` | `3000` | Number of training episodes |
| `test_episodes` | `100` | Number of evaluation episodes |
| `agent_hidden_dim` | `32` | Hidden layer width for each agent MLP |
| `mixer_hidden_dim` | `32` | Hidden layer width for the mixer hypernetwork |
| `mixer_learning_rate` | `0.001` | Learning rate for the mixing network |
| `device` | `"auto"` | Accepted values: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`. `"auto"` tries CUDA, then MPS, then CPU. |
| `replay_capacity` | `10000` | Max number of transitions stored in replay memory. |
| `batch_size` | `64` | Number of transitions sampled per gradient update. |
| `learning_starts` | `256` | Minimum replay size before starting gradient updates. |
| `train_every` | `1` | Perform one training update every N environment steps. |
| `target_update_mode` | `"hard"` | Target sync mode: `"hard"` (periodic copy) or `"soft"` (Polyak averaging). |
| `target_update_interval` | `200` | In `"hard"` mode, copy online -> target every N gradient steps. |
| `tau` | `0.01` | In `"soft"` mode, Polyak coefficient for target updates. |

The NN variant is intentionally additive: `slime_qmix.py` still uses the tabular QMIX implementation, while `slime_qmix_nn.py` provides the neural version.

For a deeper explanation of the Alternative C design decisions, see `QMIX_NN_ALTERNATIVE_C_EXPLANATION.md`.

**Selected design options for QMIX-NN:**
- **Option C**: replay buffer + mini-batch + target networks (deep-RL training path)
- **Option A**: keep `.npy` checkpoint compatibility with the current `Logger` via model pack/unpack
- **Double-Q style**: online network selects next actions, target network evaluates them

**Example: training and evaluation**

```bash
python slime_qmix_nn.py --train True --random_seed 99
python slime_qmix_nn.py --train False --qtable_path ./runs/weights/qmix_nn_train_weights_*.npy --random_seed 99
```

**Example: force GPU (if available in your PyTorch install)**

Set `"device": "cuda"` in `agents/QMIXLearningNN/config/learning-params.json`, then run:

```bash
python slime_qmix_nn.py --train True --random_seed 99
```

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
|`/agents/QMIXLearningNN/config/learning-params.json` |	Contains neural-network QMIX parameters and model sizes.|
|`/agents/QMIXLearningNN/config/logger-params.json` |	Provides run metadata naming defaults for QMIX-NN experiments.|

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