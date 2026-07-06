# Hybrid Execution Strategy for QMIX/MAPPO (EpyMARL)

## Overview

The QMIX and MAPPO runners support a **hybrid execution strategy** that automatically selects between two execution methods based on availability and configuration:

1. **Direct Python API** (preferred) — Imports and calls EpyMARL functions directly
2. **Subprocess-based** (fallback) — Spawns EpyMARL via configurable command templates

This guide explains how the system works, how to configure it for your installation, and when each method is used.

---

## Architecture

### Execution Flow

```
User runs: python slime_qmix.py --train True --random_seed 10
                    ↓
    slime_qmix.py loads learning_params.json
                    ↓
    agents/utils/epymarl_launcher.py:run_epymarl_command()
                    ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
Try Direct API              Fallback to Subprocess
(if use_direct_api=true      (if direct API returns None
 and EpyMARL available)       or force_subprocess=true)
    ↓                                   ↓
Success: return              Success: return
{method: "direct_api",       {method: "subprocess",
 ...}                         ...}
```

### Key Components

**epymarl_launcher.py** — Central dispatch logic
- `_try_direct_api_call()` — Attempts to use direct Python API
- `run_epymarl_command()` — Main function, tries direct API first, then subprocess

**qmix.py / mappo.py** — Algorithm wrappers (stateless)
- Simple delegates to `run_epymarl_command()`
- Return result dict with execution metadata

**slime_epymarl.py** — CLI orchestration
- Parses arguments (matches CoQL CLI exactly)
- Handles seed iteration and experiment batching
- Reports which execution method was used

---

## Configuration

### Learning Config Structure

Each algorithm has a learning config at `agents/{Algorithm}Learning/config/learning-params.json`:

```json
{
    "train_episodes": 3000,
    "test_episodes": 100,
    "epymarl": {
        "use_direct_api": true,
        "force_subprocess": false,
        "command_template_train": "__SET_ME__",
        "command_template_eval": "__SET_ME__"
    }
}
```

#### Configuration Options

| Field | Type | Default | Behavior |
|-------|------|---------|----------|
| `use_direct_api` | bool | `true` | If `true`, attempt direct Python API first |
| `force_subprocess` | bool | `false` | If `true`, skip direct API and use only subprocess |
| `command_template_train` | str | `"__SET_ME__"` | Command template for training (subprocess only) |
| `command_template_eval` | str | `"__SET_ME__"` | Command template for evaluation (subprocess only) |

### Decision Matrix

When `run_epymarl_command()` is called:

```
┌─────────────────────────────────────────────┐
│ Is force_subprocess = true?                 │
├─────────────────────────────────────────────┤
│ YES → Skip direct API, use subprocess       │
│ NO  → Try direct API first (see below)      │
└─────────────────────────────────────────────┘
                    ↓ (NO)
┌─────────────────────────────────────────────┐
│ Is use_direct_api = true?                   │
├─────────────────────────────────────────────┤
│ YES → Attempt direct API                    │
│ NO  → Skip to subprocess                    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Did direct API return a result?             │
├─────────────────────────────────────────────┤
│ YES → Return result (method="direct_api")   │
│ NO  → Fall through to subprocess            │
└─────────────────────────────────────────────┘
                    ↓ (NO)
┌─────────────────────────────────────────────┐
│ Validate command_template_train/eval        │
├─────────────────────────────────────────────┤
│ Missing → Raise ValueError                  │
│ Found   → Execute subprocess (method=...)   │
└─────────────────────────────────────────────┘
```

---

## Setup Examples

### A. EpyMARL Installed as Python Package

If you installed EpyMARL with `pip install pymarl` or similar:

**Config:**
```json
{
    "epymarl": {
        "use_direct_api": true,
        "force_subprocess": false,
        "command_template_train": "__SET_ME__",
        "command_template_eval": "__SET_ME__"
    }
}
```

**Run:**
```bash
python slime_qmix.py --train True --random_seed 10
# Output: [EXECUTION METHOD] DIRECT_API
```

**Behavior:**
- Direct API is attempted
- If `pymarl` can be imported, it's used directly
- Command templates are ignored
- Lowest latency, best error reporting

---

### B. EpyMARL CLI-Only Installation

If EpyMARL is installed as a standalone tool or script:

**Config:**
```json
{
    "epymarl": {
        "use_direct_api": false,
        "force_subprocess": true,
        "command_template_train": "cd /path/to/epymarl && python run.py --config={algorithm} --env-config=slime with seed={seed} env_args.env_params_path={env_params_path}",
        "command_template_eval": "cd /path/to/epymarl && python run.py --config={algorithm} --env-config=slime with seed={seed} env_args.env_params_path={env_params_path} checkpoint_path={run_tag}"
    }
}
```

**Run:**
```bash
python slime_mappo.py --train True --random_seed 10
# Output: [EXECUTION METHOD] SUBPROCESS
# Command: cd /path/to/epymarl && python run.py --config=mappo ...
```

**Behavior:**
- Subprocess is used directly (no direct API attempt)
- Template placeholders are substituted
- EpyMARL output is captured and printed

---

### C. Hybrid Setup (Recommended)

Use defaults and let the system choose:

**Config:**
```json
{
    "epymarl": {
        "use_direct_api": true,
        "force_subprocess": false,
        "command_template_train": "cd /path/to/epymarl && python run.py --config={algorithm} ...",
        "command_template_eval": "cd /path/to/epymarl && python run.py --config={algorithm} ..."
    }
}
```

**Run:**
```bash
python slime_qmix.py --train True --random_seed 10
```

**Behavior:**
- If direct API available (EpyMARL installed), use it
- Otherwise fall back to subprocess
- Provides maximum flexibility

---

## CLI Usage (Identical to CoQL)

All options mirror `slime_coql.py` and`slime_iql.py`:

### Single Run
```bash
python slime_qmix.py --train True --random_seed 99
```

### Multi-Seed Repetition
```bash
python slime_qmix.py --train True --random_seeds 10 20 30
```

### Sequential Experiments
```bash
python slime_mappo.py --train True --experiments_dir experiments
```

### Combined
```bash
python slime_qmix.py --train True --experiments_dir experiments --random_seeds 10 20 30
```

### Evaluation
```bash
python slime_qmix.py --train False --random_seed 99
```

---

## Template Placeholders (Subprocess Only)

When using subprocess mode, the following placeholders are substituted in command templates:

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{algorithm}` | Algorithm name | `qmix` or `mappo` |
| `{seed}` | Random seed | `10`, `20`, `30` |
| `{env_params_path}` | Abs path to env config | `/home/.../env-params.json` |
| `{learning_params_path}` | Abs path to learning config | `/home/.../learning-params.json` |
| `{logger_params_path}` | Abs path to logger config | `/home/.../logger-params.json` |
| `{run_tag}` | Experiment ID + seed | `seed_10`, `exp_5_seed_20` |

### Template Example
```bash
# Before substitution:
"command_template_train": "python run.py --algo {algorithm} --seed {seed} --env {env_params_path}"

# After substitution (seed=10):
# python run.py --algo qmix --seed 10 --env /full/path/env-params.json
```

---

## Debugging

### Check Execution Method
The console output always shows which method was used:

```bash
python slime_qmix.py --train True --random_seed 10

# Output:
# [EXECUTION METHOD] DIRECT_API
# or
# [EXECUTION METHOD] SUBPROCESS
```

### Inspect Commands
For subprocess execution, the full command is printed:

```
Command: cd /path/to/epymarl && python run.py --config=qmix ...
```

### Enable Verbose Logging
The launcher prints debug messages when direct API fails:

```
[DEBUG] Direct API unavailable (No module named 'pymarl'), falling back to subprocess
```

### Check Configuration
Verify your learning config is valid JSON:

```bash
python -c "import json; json.load(open('agents/QMIXLearning/config/learning-params.json'))"
```

---

## Result Format

Both execution methods return a result dict:

```python
{
    "method": "direct_api" | "subprocess",  # Execution method used
    "command": str,                          # Executed command or method string
    "returncode": int,                       # Exit code (0 = success)
    "stdout": str,                           # Standard output
    "stderr": str,                           # Standard error
}
```

### Example Direct API Result
```python
{
    "method": "direct_api",
    "command": "direct_api:qmix",
    "returncode": 0,
    "stdout": "[Direct API] Executed qmix train on /path/env-params.json with seed 10",
    "stderr": "",
}
```

### Example Subprocess Result
```python
{
    "method": "subprocess",
    "command": "cd /path/to/epymarl && python run.py --config=qmix --seed 10 ...",
    "returncode": 0,
    "stdout": "Training started...\n[Episode 100] ...",
    "stderr": "",
}
```

---

## Testing

All hybrid execution logic is tested in `tests/test_epymarl_launcher.py`:

```bash
python -m unittest tests.test_epymarl_launcher -v
```

Tests verify:
- Missing templates raise errors
- Placeholders are substituted correctly
- Direct API is attempted when configured
- Subprocess is used when direct API unavailable
- `force_subprocess=true` bypasses direct API

All 5 launcher tests + 14 others (19 total) pass.

---

## Tips & Best Practices

1. **Start with defaults** — Use `use_direct_api: true` and `force_subprocess: false` for maximum flexibility

2. **Provide fallback templates** — Always include `command_template_train/eval` even if using direct API, for compatibility

3. **Use absolute paths** — All paths in templates should be absolute to avoid working directory issues

4. **Test first** — Run with `--random_seeds 10` to validate setup before large experiments

5. **Capture logs** — Redirect subprocess output in templates:
   ```json
   "command_template_train": "cd /path && python run.py ... >> {run_tag}_train.log 2>&1"
   ```

6. **Monitor method** — Check console output to confirm which execution method is being used

7. **Consistency** — Keep config structure identical across QMIX and MAPPO for maintainability

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ValueError: missing EpyMARL command template` | No templates set | Fill in `command_template_train/eval` or enable direct API |
| `No module named 'pymarl'` | EpyMARL not installed | Install via `pip`, or set `force_subprocess=true` |
| `returncode: 1` | EpyMARL execution failed | Check template syntax and paths; check `stderr` output |
| `force_subprocess=true` ignored | Direct API still attempted | Set `use_direct_api: false` explicitly |
| Placeholder not substituted | Typo in template | Use exact placeholder names: `{algorithm}`, `{seed}`, etc. |

---

## Future Enhancements

Possible extensions to the hybrid system:

1. **Direct environment creation** — Pass environment params directly to EpyMARL functions instead of via files
2. **Policy checkpointing** — Return trained models directly instead of file paths
3. **Metrics streaming** — Capture episode rewards/metrics during training without file I/O
4. **Multi-process support** — Parallelize multiple seeds within a single process
5. **Ray integration** — Use Ray Tune for distributed training via direct API

---

## Summary

The hybrid execution strategy balances **flexibility** (subprocess templates) with **efficiency** (direct API), automatically choosing the best method for your installation. All CLI commands are identical across IQL, CoQL, QMIX, and MAPPO for consistency.


