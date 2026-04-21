# Asymptotically Optimal Localized Distributed Beamforming

This repository contains a publication-style draft and a reproducible simulation stack for localized distributed beamforming and power allocation in cooperative multi-cell downlink systems. Each base station exchanges optimization variables only with neighbors inside a cooperation radius `rho`, and the resulting performance approaches a globally coordinated benchmark as the neighborhood expands.

## Motivation

Coordinated multi-point transmission can deliver large gains, but full coordination requires network-wide CSI exchange and tight synchronization. In many deployments, interference is spatially local because channel strength decays with distance, which suggests that useful coordination should also be local.

## Key Idea

Each base station only exchanges information with nearby cooperating stations inside radius `rho`.

As `rho` increases:

- signaling cost increases locally
- achieved weighted sum-rate approaches the globally coordinated solution

The repository studies this tradeoff with a localized distributed beamforming and power allocation algorithm, a centralized benchmark, and reproducible figures.

## Repository Contents

- [paper/main.tex](/Users/yosia/Desktop/ideas/bs_coop/paper/main.tex): IEEE-style conference paper draft
- [paper/refs.bib](/Users/yosia/Desktop/ideas/bs_coop/paper/refs.bib): bibliography, including the mandatory `Jungnickel08` citation
- [DERIVATION_AND_SIMULATION_CHECK.md](/Users/yosia/Desktop/ideas/bs_coop/DERIVATION_AND_SIMULATION_CHECK.md): GitHub-friendly theorem derivation and simulation verdict
- [code/experiments.py](/Users/yosia/Desktop/ideas/bs_coop/code/experiments.py): Monte Carlo experiment runner
- [code/plot_results.py](/Users/yosia/Desktop/ideas/bs_coop/code/plot_results.py): publication-figure generator
- [results/raw](/Users/yosia/Desktop/ideas/bs_coop/results/raw): CSV outputs from experiments
- [results/figs](/Users/yosia/Desktop/ideas/bs_coop/results/figs): generated PNG and PDF figures

## How to Run Experiments

Install dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Run the simulation suite:

```bash
python code/experiments.py
python code/plot_results.py
```

Compile the paper:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Expected Results

The generated figures should show that:

- weighted sum-rate improves monotonically with the cooperation radius
- the gap to the global benchmark shrinks as `rho` grows
- localized coordination scales more gently in runtime than full coordination
- signaling cost rises with neighborhood size
- larger path-loss exponents make locality more effective, so the gap closes faster as `rho` increases

Small and medium-sized neighborhoods should recover most of the global coordination gain while keeping coordination overhead moderate. On individual nonconvex instances, finite-iteration behavior need not be perfectly monotone across every `rho`, so the repository reports averaged true-objective trends and larger-network locality studies.

## Citation

If you use this repository, cite the paper draft in this repository and the foundational predecessor work:

- Volker Jungnickel et al., "Distributed Base Station Cooperation via Block-Diagonalization and Dual-Decomposition," Proc. IEEE Globecom, 2008.
