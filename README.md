# mrl-trace

A Memristive Device Transient as a Hardware Eligibility Trace for Three-Factor Reinforcement Learning.

This repository contains the reference implementation for the manuscript *"A Memristive Device Transient as a Hardware Eligibility Trace for Three-Factor Reinforcement Learning"*. 

The package models the SiOₓ device transient to serve as a hardware eligibility trace for spiking neural networks and reinforcement learning tasks. It utilizes `torch`, `snntorch`, and standard scientific libraries.

## Project Structure

- `src/mrl_trace/`: Core Python package containing models for devices, neurons, learning rules, and various simulated tasks.
- `experiments/`: Jupyter notebooks containing the reproducible experiments presented in the manuscript.

## Core Modules

- **device**: The fitted SiOₓ transient and the `TransientGate` eligibility-trace generator.
- **neurons**: Leaky Integrate-and-Fire (LIF) neuron implementations.
- **learning**: Signed coincidence kernel and three-factor reward-modulated updates.
- **bandit / maze / probselect**: Simulated environments to test closed-loop RL performance.
- **selectivity**: Tasks that read out the transient's tuning to inter-cue delays.
- **extensions**: Multi-timescale memory, short-term consolidation, and device-TD extensions.

## Installation

Ensure you have Python 3.10+ installed. You can install this package in editable mode via pip:

```bash
pip install -e .
```

Dependencies such as `numpy`, `torch`, `snntorch`, `matplotlib`, and `scipy` will be automatically installed.

## Reproducing Experiments (Publication Claims)

The `experiments/` directory contains numbered Jupyter notebooks corresponding to the primary claims of the study:
- `01_trace_and_credit_window.ipynb`: Demonstrates the device transient characterization and the resulting asymmetric credit assignment window.
- `02_closed_loop_rl.ipynb`: Validates system-level closed-loop RL performance (bandit, maze, probabilistic selection tasks).
- `03_deep_local_learning.ipynb`: Extends the rule to deep local learning in multi-layer perceptrons.
- `04_temporal_selectivity.ipynb`: Shows temporal selectivity and tuning to inter-cue delays.
- `05_biological_grounding.ipynb`: Connects the hardware trace to biological observations (e.g., dopamine).
- `06_extensions.ipynb`: Covers multi-timescale memory, short-term consolidation, and device-TD extensions.

### Replay Mode vs. Full-Sweep Mode

By default, the notebooks run in **replay mode** using pre-computed outputs and cached grids stored in `data/results/`. These cached grids constitute the publication-scale evidence. Replaying them executes cleanly in seconds (the expected runtime for `REPRODUCE_ALL.ipynb` in replay mode is under 1 minute). 

Running the full publication-scale sweeps from scratch (e.g., repeating all hardware simulations across many random seeds) is **not** performed by default due to significant compute requirements and the need for external data sets.

For sanity checks and reproduction paths, you can run reduced-seed, smaller-scale versions of the tasks. A `REPRODUCE_ALL.ipynb` notebook is also provided to automatically run all topic notebooks end-to-end.