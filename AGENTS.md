# Repository Guidance for Coding Agents

- Keep the paper notation and the Python implementation aligned. If a variable name changes in the manuscript, update the code comments and README when relevant.
- Preserve deterministic seeds for all experiments.
- Prefer adding new experiments as separate functions in `code/experiments.py` and new plots as separate functions in `code/plot_results.py`.
- Do not remove the mandatory `Jungnickel08` citation from the paper or bibliography.
- Keep generated artifacts in `results/` and source files in `paper/` or `code/`.
