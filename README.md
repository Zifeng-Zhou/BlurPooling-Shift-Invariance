# Does Blur Pooling Help a CIFAR-10 Classifier Trained on Shifted Images?

Rice University · ELEC 576 / COMP 576 · Fall 2026  
Haotian Si (team captain), Hongbo Yu, Zifeng Zhou (zz188)

The experiment crosses both pooling sites in a compact LeNet with translation-only augmentation. The primary comparison pairs the two augmented models at three seeds. See [the frozen protocol](docs/protocol.md), [run status](docs/STATUS.md) and [handoff](docs/HANDOFF.md).

## Reproduce on a CUDA machine

Use Python 3.11 and a CUDA 12.8-compatible NVIDIA driver. These PowerShell commands assume the repository root. A fresh clone needs the development branch.

```powershell
git clone --branch codex/blur-pooling-final https://github.com/Zifeng-Zhou/COMP576-Final.git
cd COMP576-Final
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
nvidia-smi
& $py -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.ones(2,device='cuda').sum().item())"
& $py data.py
& $py -m unittest -v test_protocol
& $py gpu_check.py
& $py train.py smoke --condition D
& $py train.py timing --condition D
& $py train.py all
```

The recorded machine used a project-local venv with read-only access to an existing compatible PyTorch installation. No global package was modified. A valid cached official archive can be copied to `data/cifar-10-python.tar.gz` before preparation. The archive MD5 must equal `c58f30108f718f92721af3b95e74349a`. Training and validation indices index the concatenation of official data_batch_1 through data_batch_5. The first official test label enters the analysis only after the code, configuration and primary comparison have been fixed.

## Historical audit and final evaluation

Set the path once to the extracted **assignment2** directory containing the course image ZIP and outputs_part1. These private materials are needed only for the pilot and overlap audit.

```powershell
$assignment = '../../work/COMP576_Codex_Handoff/materials/assignment2'
& $py pilot.py $assignment
& $py overlap.py $assignment
& $py evaluate.py
& $py analyze.py
& $py examples.py
```

Without course files, the official full-test evaluation still runs. Evaluation automatically uses the 9,000 committed subset indices when the local audit is absent. If both are present, their contents must agree. Sample IDs in prediction files are zero-based within the official test split. Column order is left, right, up, down. All raw metrics are fractions in `results/evaluation.csv`. Paired tables use percentage points. Cross-seed dispersion in the paired mean table uses sample SD (denominator 2).

## Resume

```powershell
& .\.venv\Scripts\python.exe train.py all
# Or continue a single run, for example D at seed 2
& .\.venv\Scripts\python.exe train.py single --condition D --seed 2
```

Each epoch commits the model, optimizer, learning-rate scheduler, selected weights, RNG state and history together in an atomic last.pt. A mid-epoch interruption replays the unfinished epoch from its start. The batch order and augmentation stream are recomputed from epoch and seed. A completed run is verified against four hashes before reuse. Restore the original source/config and environment for exact continuation. Repeating from random initialization uses a fresh clone or a separate copy with an empty artifacts directory. Device and library changes can affect numerical results.

## Files

| Location | Contents |
|---|---|
| results/runs/ | Per-run configs, 60-epoch logs, selected epochs and local weight hashes |
| artifacts/runs/ | Local training evidence, including best.pt and last.pt |
| artifacts/predictions/ | Per-image compressed predictions, indexed by results/predictions_manifest.json |
| artifacts/overlap/ | Content-match audit and nonpilot IDs |
| results/ | Data provenance, held-out evaluation, paired contrasts and decision |
| report/ | IEEE LaTeX, BibTeX, figures and PDF |
| poster/ | Editable PowerPoint poster and PDF |
| presentation/ | English script, Chinese explanation and questions |

To compile the paper from its directory, run `tectonic paper.tex` (or use LaTeX, BibTeX, LaTeX twice with IEEEtran installed). Edit the supplied one-slide PowerPoint directly and export as PDF at its original page size. Plot source is in analyze.py. The data and large local evidence are excluded from Git.
