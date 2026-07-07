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

## Reproducing Experiments

The `experiments/` directory contains numbered Jupyter notebooks corresponding to different parts of the study:
- `01_trace_and_credit_window.ipynb`
- `02_closed_loop_rl.ipynb`
- `03_deep_local_learning.ipynb`
- `04_temporal_selectivity.ipynb`
- `05_biological_grounding.ipynb`
- `06_extensions.ipynb`

A `REPRODUCE_ALL.ipynb` notebook is also provided to run all experiments end-to-end.