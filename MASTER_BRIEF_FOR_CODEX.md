# Project Master Brief for Codex

## Title

Asymptotically Optimal Localized Distributed Beamforming via Expanding Cooperation Neighborhoods

## Short Description

Create a full research repository containing:

1. IEEE-style LaTeX paper
2. Python simulation code
3. Reproducible figures
4. README
5. Clean project structure

The paper proposes a localized distributed joint beamforming and power allocation algorithm for cooperative multi-cell downlink systems. Each base station coordinates only with nearby neighbors inside radius `rho`. As `rho` increases, performance approaches the globally coordinated benchmark under spatial-decay assumptions.

This repository should be publication-grade.

## Mandatory Citation to Prior Work

The new paper must explicitly cite the following prior paper as foundational motivation and predecessor work:

Yosia Hadisusanto et al.,
"Distributed Base Station Cooperation via Block-Diagonalization and Dual-Decomposition",
2008.

PDF:
[Distributed Base Station Cooperation via Block-Diagonalization and Dual-Decomposition](https://www.researchgate.net/profile/Volker-Jungnickel/publication/224356509_Distributed_Base_Station_Cooperation_via_Block-Diagonalization_and_Dual-Decomposition/links/550436670cf24cee39fe1e76/Distributed-Base-Station-Cooperation-via-Block-Diagonalization-and-Dual-Decomposition.pdf)

## Positioning of Citation

State clearly:

This prior work introduced a distributed cooperative multi-cell framework using:

1. Block-diagonalization precoding
2. Dual decomposition for distributed power allocation
3. Per-antenna / local power constraint handling
4. Early distributed base-station cooperation architecture

The new paper is a direct successor that extends this line of work by replacing fixed block-diagonalization with localized joint beamforming and distributed power allocation.

## Repository Structure

Create:

```text
root/
  README.md
  LICENSE
  .gitignore

paper/
  main.tex
  abstract.tex
  intro.tex
  related_work.tex
  system_model.tex
  algorithm.tex
  theorem.tex
  simulations.tex
  conclusion.tex
  refs.bib
  figs/

code/
  config.py
  channels.py
  utils.py
  localized_algorithm.py
  global_benchmark.py
  experiments.py
  plot_results.py

results/
  raw/
  figs/
```

## README Content

Title:
Asymptotically Optimal Localized Distributed Beamforming

Sections:

1. Motivation
2. Key Idea
3. Repository Contents
4. How to Run Experiments
5. Expected Results
6. Citation

Key idea:

Each base station only exchanges information with nearby cooperating stations inside radius `rho`.

As `rho` increases:

- signaling cost increases locally
- achieved weighted sum-rate approaches globally coordinated solution

## LaTeX Paper Requirements

Use IEEEtran conference format.

`main.tex` should include:

```latex
\documentclass[conference]{IEEEtran}
```

Include:

- abstract
- intro
- related_work
- system_model
- algorithm
- theorem
- simulations
- conclusion

Use BibTeX references.

Compile cleanly.

## Abstract Content

We study joint beamforming and power allocation in cooperative multi-cell downlink networks under local signaling constraints. Instead of globally coordinated precoding requiring network-wide CSI exchange, we propose a localized distributed framework in which each base station cooperates only with neighboring stations inside an expanding cooperation neighborhood. The method combines local beamformer optimization, distributed power control, and neighborhood interference pricing. We prove that the localized optimum approaches the globally coordinated optimum as the cooperation radius grows under standard spatial-decay assumptions. Simulations show that small neighborhoods capture most of the global coordination gain with much lower signaling cost.

## Introduction Content

Topics:

- Coordinated multipoint and network MIMO improve rates but require heavy coordination
- Global CSI exchange is expensive
- Interference is spatially local due to path loss
- Therefore localized cooperation is natural
- Existing distributed beamforming often lacks explicit asymptotic optimality guarantees
- This paper provides:
  1. localized optimization model
  2. distributed beamforming algorithm
  3. theorem showing convergence toward global optimum as `rho` increases

Contribution bullets:

1. Localized cooperative beamforming formulation
2. Distributed joint beamforming + power control algorithm
3. Optimality gap theorem
4. Simulation benchmarks

Text to add to introduction:

Early work on distributed base-station cooperation considered structured linear precoding combined with distributed resource allocation. In particular, Jungnickel et al. proposed block-diagonalization together with dual decomposition for cooperative multi-cell transmission under local power constraints. While that framework demonstrated the potential of distributed coordination, it relied on fixed interference-nulling precoders and did not characterize how localized cooperation approaches the fully coordinated optimum as cooperation neighborhoods expand. The present work builds directly on that direction by enabling localized joint beamforming and power control with asymptotic optimality guarantees.

## System Model Content

System:

- `B` base stations
- BS `b` has `M` antennas
- user set `\mathcal{U}_b`

Variables:

`\vect{w}_{b,u}` complex beamforming vector

Transmit signal:

```latex
\vect{x}_b = \sum_u \vect{w}_{b,u} s_u
```

Received user `k`:

```latex
y_k = \sum_b \vect{h}_{k,b}^{\mathrm{H}} \vect{x}_b + n_k
```

SINR:

```latex
\mathrm{SINR}_k =
\frac{
\left|\vect{h}_{k,b(k)}^{\mathrm{H}} \vect{w}_{b(k),k}\right|^2
}{
\text{intra-cell interference}
+ \text{inter-cell interference}
+ \sigma_k^2
}
```

Rate:

```latex
R_k = \log_2(1 + \mathrm{SINR}_k)
```

Global objective:

```latex
\max_{\mathbf{W}} F(\mathbf{W}) = \sum_k \alpha_k R_k(\mathbf{W})
```

subject to:

```latex
\sum_u \|\vect{w}_{b,u}\|^2 \le P_b
```

Optional per-antenna constraints.

## Localization Model

Define cooperation neighborhood:

```latex
\mathcal{C}_b(\rho) = \{ c : \mathrm{distance}(b,c) \le \rho \}
```

Only BSs in `\mathcal{C}_b(\rho)` exchange variables.

Localized interference keeps only coupling terms from `\mathcal{C}_b(\rho)`.

Define localized objective:

```latex
F^{(\rho)}(\mathbf{W}) = \sum_k \alpha_k R_k^{(\rho)}(\mathbf{W})
```

## Algorithm Section Content

Name:
LDBPA = Localized Distributed Beamforming and Power Allocation

Use iterative updates.

Variables:

- beamformers `\vect{w}`
- receiver filters `g`
- MSE weights `v`
- power duals `\lambda`
- interference prices `\pi`

Iteration:

Step 1:
Each user computes MMSE receiver:

```latex
g_k =
\frac{\text{desired signal}}{\text{total received power} + \text{noise}}
```

Step 2:
Update weight:

```latex
v_k = \frac{1}{\mathrm{mse}_k}
```

Step 3:
Each BS `b` updates beamformers:

```latex
\vect{w}_{b,u} = \mathbf{A}_b^{-1} \vect{d}_{b,u}
```

where

```latex
\mathbf{A}_b =
\sum_k \alpha_k v_k |g_k|^2 \vect{h}_{k,b} \vect{h}_{k,b}^{\mathrm{H}}
+ \sum_q \lambda_{b,q} \mathbf{E}_q
+ \sum_{\text{protected users } k} \pi_k \vect{h}_{k,b} \vect{h}_{k,b}^{\mathrm{H}}
```

Step 4:
Power dual update:

```latex
\lambda_{b,q} = \max(0, \lambda_{b,q} + \gamma (\text{used power} - \text{power limit}))
```

Step 5:
Interference price update:

```latex
\pi_k = \max(0, \pi_k + \eta (\text{local interference} - \text{budget}))
```

Repeat until convergence.

Only neighbor communication inside radius `rho`.

## Theorem Section Content

State assumptions:

A1. Path loss decay:

```latex
\|\vect{h}_{k,b}\|^2 \le C (1 + d(k,b))^{-\alpha}, \quad \alpha > 2
```

A2. Feasible beamformer norms bounded.

A3. Objective Lipschitz in aggregate interference.

Definitions:

`F^{\star}` = global optimum of full problem

`F_{\rho}^{\star}` = optimum of localized radius `rho` problem

Main theorem:

There exists constant `K > 0` such that

```latex
0 \le F^{\star} - F_{\rho}^{\star} \le K \rho^{2-\alpha}
```

Hence:

```latex
\lim_{\rho \to \infty} F_{\rho}^{\star} = F^{\star}
```

Practical algorithm theorem:

If algorithm returns `\mathbf{W}_{\rho}^{\dagger}` satisfying local gap `\delta_{\rho}`:

```latex
F^{(\rho)}(\mathbf{W}_{\rho}^{\star}) - F^{(\rho)}(\mathbf{W}_{\rho}^{\dagger}) \le \delta_{\rho}
```

then

```latex
F^{\star} - F(\mathbf{W}_{\rho}^{\dagger})
\le K \rho^{2-\alpha} + \delta_{\rho}
```

Proof sketch:

1. Bound far-field interference tail
2. Use Lipschitz continuity of rates
3. Compare localized and global objectives
4. Use optimality of localized solution

## Related Work Content

Discuss:

- Coordinated multipoint
- Network MIMO
- Distributed WMMSE
- Interference pricing
- Clustered cooperation
- Cell-free massive MIMO

Position paper as:
optimization + locality scaling theorem

Text to add to related work:

One of the earliest distributed formulations of cooperative base-station transmission employed block-diagonalization precoding together with dual decomposition for distributed power allocation under per-antenna constraints [Jungnickel08]. That work established an important decomposition principle for cooperative cellular systems. In contrast, the present paper removes the need for fixed block-diagonalization structures and develops a localized beamforming architecture whose performance provably approaches the globally coordinated benchmark as the cooperation radius increases.

Text to add to contributions:

This paper generalizes the distributed decomposition philosophy of Jungnickel08 from fixed precoding plus power loading to fully adaptive localized joint beamforming and power allocation.

## Simulation Section Content

Implement Python experiments.

Topology:

- 7-cell hexagonal layout
- random BS PPP layout optional

Users:
1 or 2 users per BS

Channels:
Rayleigh fading with path loss exponent `alpha`

Compare:

1. Localized algorithm `rho=1`
2. `rho=2`
3. `rho=3`
4. Global benchmark
5. Non-cooperative baseline

Metrics:

- weighted sum-rate
- runtime
- signaling messages
- convergence iterations

Expected plots:

1. Sum-rate vs `rho`
2. Gap to global optimum vs `rho`
3. Runtime vs number of BS
4. Signaling cost vs `rho`

Need figure export as PNG and PDF.

## Python Code Requirements

Use:

- `numpy`
- `scipy`
- `matplotlib`

Files:

`channels.py`

- generate path loss
- fading channels
- layouts

`localized_algorithm.py`

- implement iterative algorithm

`global_benchmark.py`

- centralized benchmark using `scipy.optimize` or a tractable centralized approximation

`experiments.py`

- run Monte Carlo trials

`plot_results.py`

- create publication figures

Use random seeds for reproducibility.

## Coding Quality

- clean modular code
- comments
- deterministic seeds
- no dead code
- save figures automatically
- save CSV results

## BibTeX Entry to Add to refs.bib

```bibtex
@inproceedings{Jungnickel08,
  author    = {Yosia Hadisusanto and others},
  title     = {Distributed Base Station Cooperation via Block-Diagonalization and Dual-Decomposition},
  booktitle = {Proc. IEEE Globecom},
  year      = {2008}
}
```

If exact venue metadata is later verified, replace placeholder fields with accurate bibliographic information.

## Strategic Positioning

Frame the new paper as:

- 2008 predecessor: Fixed BD precoding + distributed dual power allocation
- New paper: Localized adaptive beamforming + distributed power control + asymptotic approach to global optimum

## Commit Plan

Commit 1:
Initialize repo structure

Commit 2:
Add IEEE paper draft

Commit 3:
Implement simulation framework

Commit 4:
Generate figures

Commit 5:
Polish README and final build

## Final Deliverables

Need a working repo where the following commands run:

```bash
python code/experiments.py
python code/plot_results.py
```

and:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Important Style Notes

- Write like a serious IEEE paper
- No hype
- Mathematically clean
- Use precise notation
- Keep proofs concise
- Make code runnable

## Bonus Tasks

If time permits:

1. Add `AGENTS.md` for coding agents
2. Add arXiv version
3. Add appendix with extended proofs
4. Add notebook demo

## Do Not Omit This Reference

`Jungnickel08` is mandatory and should be treated as foundational prior art in the manuscript narrative.
