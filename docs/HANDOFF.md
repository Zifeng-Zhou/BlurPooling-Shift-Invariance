# Final project handoff

Pre-presentation version, November 24, 2026. All 12 formal runs, the recorded analyses and the presentation materials are complete. Human contribution statements and subsequent course feedback remain open.

The [pre-submission rubric review](SUBMISSION_REVIEW.md) maps each grading item to the revised paper or poster and records the checkpoint replay, table and figure checks, and outstanding course actions.

## Submission materials

| Deliverable | Repository path |
|---|---|
| IEEE report | report/paper.pdf |
| Editable paper and references | report/paper.tex, report/references.bib, report/figures/ |
| Editable one-slide poster and submission PDF | poster/poster.pptx, poster/poster.pdf |
| Five-minute English script and optional three-speaker split | presentation/talk.md |
| Chinese explanation and local commands | presentation/understanding-zh.md |
| Five-minute Q&A preparation | presentation/questions.md |
| Reproduction, checks and recovery | README.md, docs/protocol.md, docs/STATUS.md |

The report uses the IEEEtran conference class. The five-page US Letter PDF has four full pages followed by concluding text, code availability, roles and references on page five. The poster PDF has one 36 × 24 inch page. That adjustable landscape size is a project choice pending course confirmation. The PowerPoint contains editable text, a native experiment table and a native chart with its six plotted values in an embedded workbook. The final PDFs were rasterized and inspected. The talk has about 571 spoken words; rehearse with the planned handoffs to stay within five minutes.

## Findings

Under reflected one-pixel shifts, the augmented standard and blur classifiers averaged 15.677% and 10.036% flips. D minus B was −5.278, −5.950 and −5.695 pp at seeds 0, 1 and 2. Mean clean and shifted differences were +0.037 and +0.112 pp. All three parts of the prespecified rule passed. Seed 0 lost 0.700 pp clean and 0.615 pp shifted accuracy. C minus A improved flips, clean and shifted accuracy by −12.791, +4.563 and +4.727 pp on average.

The mean-filled one-pixel flip difference was −5.866 pp. At two pixels the reflected/mean-fill differences were −2.819/−3.122 pp. At four pixels they were −0.033/−0.404 pp. The practical criterion applies to the primary endpoint. The four-pixel reflected effect varied in sign.

The course pilot reproduced 816 changes in 4,000 views, 631 correct clean labels and 2,435 correct shifted labels. All 1,000 prior course test images were identified within the official test set by content. Of 10,000 old training images, 9,999 had a unique match and one remained ambiguous. The 9,000-image nonpilot primary contrast was −5.662 pp in flips, +0.048 pp in clean accuracy and +0.108 pp in shifted accuracy. The complete official test and this subset share observations.

## Evidence and recovery

| Evidence | File or directory | Units or interpretation |
|---|---|---|
| Official archive, stats and index digests | results/split.json | Train and validation arrays are data/train.npy and data/val.npy locally |
| All runs and selected epochs | results/run_manifest.json | best.pt and last.pt SHA-256 per run |
| Configs, source versions and training curves | results/runs/{condition}{seed}/ | 60 rows per history, fractions for accuracy |
| Official and nonpilot metrics | results/evaluation.csv | 144 rows, proportions, direction-averaged scores |
| Per-seed and mean contrasts | results/paired_pp.csv, results/paired_mean_pp.csv | pp; SD across three paired seeds |
| Prespecified decision | results/decision.json | Computed from unrounded full-test data |
| Per-image predictions | artifacts/predictions/*.npz | 72 files with IDs, truth, clean and four directional predictions |
| Prediction integrity | results/predictions_manifest.json | SHA-256 and local relative paths |
| Pilot counts and evidence | results/pilot.json, artifacts/pilot/predictions.csv | Grayscale only |
| Exposure audit | results/overlap.json, artifacts/overlap/matches.csv, results/nonpilot_ids.txt | Candidate pixel error and identity status |
| Case illustrations | results/examples.json | Seed 0; first qualifying test ID under the recorded selection |
| Local weights | artifacts/runs/{A,B,C,D}{0,1,2}/{best,last}.pt | 24 formal files; smoke/timing artifacts are stored separately |

Every formal run is complete. From the repository root, re-enter the same environment and run `.\.venv\Scripts\python.exe train.py all`. Completed evidence is checked and reused. The most recent complete epoch is the restart point for a partial run. For a clean rebuild, use a new clone and follow README.md. For final evaluation, run evaluate.py, analyze.py and examples.py in that order. The committed nonpilot ID list permits the recorded subset comparison without redistributing the course data.

Protocol checks passed on CPU. GPU replay versus ordinary SGD and epoch-boundary recovery matched weights exactly on the same device in the dedicated small tests. All 720 formal epoch logs passed checks for LR, order pairing and validation selection. Recomputing all 144 metric records from the saved predictions agreed to numerical precision. All 72 prediction hashes were verified.

## Difficulties and interpretation

The university download experienced interrupted transfers. An existing local official RGB archive passed MD5 validation and supplied the data. A small grayscale normalization fixture exposed a hard-coded channel count in a helper, corrected before formal training. No hypothesis, selection rule or formal run was removed or revised after seeing test accuracy. An initial combined Git command failed to obtain credentials in the sandbox; a standalone authorized Git push succeeded. The poster PDF was exported with the presentation renderer and verified as a single page with selectable text. Native PowerPoint opening was not verified.

There are three independent training seeds. The 40,000 directional comparisons per checkpoint do not provide 40,000 independent model repetitions. Blur altered both filtering and the effective receptive field. Cropping at the border and displacement relative to the dense head complicate any causal claim about the four-pixel result. Conclusions cover the specified compact classifier and integer shifts.

## Team and course actions

- The report's **Group Members Roles** section records the proposal assignments as a draft. Before final submission, have each member confirm what they completed, including any work they shared or took over, then revise that section. Haotian Si is team captain. Zifeng Zhou is zz188 and authored the historical Assignment 2 work. The talk's speaking order is adjustable.
- Check the current Canvas announcements for the poster size, time zone, complete availability-form link and presentation slot. The rubric snapshot lists December 2 at 11:59 pm for availability, December 8 at 5 pm for the poster, December 9 for presentations, and December 15 at 11:59 pm for the report. The form was TBC and Step 4 contains a 2025 date in a 2026 document.
- The captain submits the availability form, checks grader access to the development branch, uploads the poster PDF to Gathertown first and Canvas second. All members should log in at least 15 minutes early.
- After the presentation, record actual TA/instructor feedback, update the report and contribution section, compile and inspect the new PDF, then complete the course submission. No course platform submission has been made.
