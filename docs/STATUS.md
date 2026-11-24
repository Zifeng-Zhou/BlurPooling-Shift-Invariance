# Experiment status
The source inventory matched all nine SHA-256 digests in the local handoff. The repository had no advertised refs, commits, README, AGENTS.md or teammate changes at clone time. Development branch: codex/blur-pooling-final.

**Course requirements.** IEEE conference paper (4–6 pages before references), one-page poster PDF submitted by the captain to Gathertown then Canvas, five minutes for the presentation and five for questions. Code availability and figures are graded.

**Proposal commitments.** Two pooling sites, four conditions, three paired seeds, 60 epochs, validation selection, primary reflected one-pixel test, mean-fill and larger translations. The old 28-pixel grayscale data and test-selected Setting C checkpoint are held separately.

**Implementation decisions.** Training seeds 0, 1, 2; stratification seed 42. Independent uniform integer offsets per axis. Full FP32. Reflect internally before fixed blur sampling, on even-index centers of the sliding maxima.

**Status.** Verified RTX 5090 with PyTorch 2.11.0+cu128, compute capability 12.0 and a CUDA tensor operation. Official RGB archive verified. All 12 runs completed 60 epochs on this GPU. All six padding/magnitude combinations were evaluated on the full and 9,000-image nonpilot sets. Five protocol tests passed. Additional CUDA replay and exact restart checks passed. See results/run_manifest.json and the per-run histories. Restore from the repository root with .venv/Scripts/python.exe train.py all. The editable report and poster, both rendered PDFs, spoken script, Chinese explanation, Q&A and handoff are complete. No training is active.

**Human confirmation.** Completed human roles, poster dimensions, schedule/time zone, availability link, grader access, and feedback after the presentation remain open. Haotian Si is team captain. The local rubric has an inconsistent Step 4 year.

## Run ledger

| Seed | A | B | C | D |
|---|---|---|---|---|
| 0 | 60/60 | 60/60 | 60/60 | 60/60 |
| 1 | 60/60 | 60/60 | 60/60 | 60/60 |
| 2 | 60/60 | 60/60 | 60/60 | 60/60 |

See [the handoff](HANDOFF.md) for findings, evidence paths, artifact checks and course actions.
