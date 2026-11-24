# Pre-submission rubric review

November 24, 2026. Reviewed against the course's November 24 rubric snapshot and the supplied Canvas poster instructions. The computational results agree with the saved validation-selected models. The report's actual-contribution requirement remains open. The point values below are the instructor's weights.

## Final report

| Rubric item | Weight | Finding and location |
|---|---:|---|
| Submission format | 5 | IEEEtran conference class, US Letter, two columns. Four complete pages precede the fifth page containing the end of the discussion, availability, roles and references. All five figures and both tables appear in the PDF. |
| Abstract | 5 | Page 1 states the practical problem, matched experiment, one-pixel flip rates, accuracy changes and four-pixel limit. The displayed values agree with the unrounded paired analysis. |
| Background and motivation | 20 | Section I and Fig. 1 explain how a one-pixel movement can move a strong feature across a pooling window. The revision distinguishes the published circular CIFAR-10 setup from our reflected and mean-filled crops. The historical grayscale pilot is identified as test-selected. The contribution is an empirical comparison for this setting. |
| Experimentation and model | 30 | Section II and Fig. 2 cover A–D, the 45,000/5,000 split, training-only statistics, both pooling stages, matched seeds, augmentation, SGD schedule, validation selection, endpoint definitions and tolerance. Section IV documents an interrupted download and a corrected normalization helper. All twelve runs completed. |
| Results | 20 | Section III, Tables I and II and Figs. 3–4 report individual seeds, validation-selected epochs, accuracy trade-offs, C−A, border and distance sensitivity, and the nonpilot analysis. Table entries agree with the independent recount at their displayed precision. Fig. 5 documents a success and a failure on identified images. |
| Discussion and conclusion | 10 | Section IV relates the result to the proposal's expectation. It distinguishes a three-seed stability result from the mixed accuracy effects and gives a testable explanation for the smaller effect at four pixels. Phase, content loss and classifier readout remain possible explanations. |
| Code availability | Prerequisite for model and results credit | Page 5 links the repository, its development branch and the reproduction README. The PDF contains all three link targets. The development branch contains the source and logs. The captain must verify access from a grader account or with the designated graders. |
| Group Members Roles | 5 | Page 5 contains the four proposal assignments under *Proposal plan (draft)*. Final credit depends on describing the work each person actually completed. All three members must confirm individual and shared contributions before final submission. |
| References | 5 | The bibliography covers prior work on shift sensitivity and blur pooling, the original course report, CIFAR-10 and the reference implementation. Added the 2009 technical report requested by the dataset authors. Citation links and references resolve in the compiled paper. |

## Poster

| Rubric item | Weight | Finding and location |
|---|---:|---|
| Title and authors | Required, ungraded | The top of the single-page poster gives the project title, all three names, team captain and Rice University. The editable slide and PDF are both 36 × 24 inches, pending the course's final size instructions. |
| Abstract | 5 | States the question in everyday language, gives the 10,000-image population, explains the four directional views and reports the two augmented flip rates. |
| Background and motivation | 30 | The truck triptych shows the original image, a rightward move and the affected pixels. The text ties the pooling-window problem to published results and asks what blur adds after shift training. |
| Experimentation and model | 30 | The editable two-by-two table now names all four conditions and reports each mean flip rate. The adjacent text states the input format, two pooling sites, displacement range, matched randomness, training duration and checkpoint selection. |
| Results | 20 | The grouped bars compare B and D seed by seed under one-pixel reflection. We checked their order, six underlying chart values and the embedded workbook against replayed predictions. The text covers both accuracy outcomes and seed 0's clean accuracy loss. |
| Conclusion | 10 | The accuracy tolerance and direction of all three flip differences are explicit. The poster gives the reflected 1-, 2- and 4-pixel reductions and the one-pixel mean-fill check. It distinguishes the completed schedule from a possible future larger-canvas experiment. |
| References | 5 | Numbered references support the motivation, blur method and CIFAR-10 data. The poster now cites the original dataset technical report. |

## Numerical and visual checks

- Reopened the official archive, verified its published checksum and checked all twelve complete run records, including the SHA-256 of each selected and restart checkpoint. Each history contains epochs 1–60. Independently selecting the earliest maximum of the clean validation correct count recovered all twelve selected epochs.
- Replayed each selected model on the 10,000 clean test images and 24 translated views per image (two border rules, three distances, four directions). A separate pad-and-crop implementation reproduced all 120,000 clean and 2,880,000 shifted class predictions using the project FP32 normalization. IDs, true labels, direction order and all 72 saved prediction-file hashes matched.
- Recounted flips and both accuracies from the individual predictions, then recomputed 144 evaluation rows, 72 paired contrasts and 24 paired summaries, including sample SD across seeds. The nonpilot index list matched the complement of the 1,000 identified historical test images. The historical prediction file yielded 816 flips, 631 correct clean and 2,435 correct shifted predictions.
- Rebuilt every report figure into a scratch directory. All five PNGs and all five rasterized figure PDFs matched the versioned figure files pixel for pixel. The supplementary case script recovered test IDs 108 and 22 and their archived predictions. The final five paper pages and the poster PDF were rendered and inspected. The one-slide PowerPoint passed chart, workbook and layout validation. Native PowerPoint opening was not part of this review.
- A diagnostic showed why the exact normalization implementation matters for borderline predictions. Dividing each channel by a Python scalar instead of the project's broadcast FP32 tensor changed normalized inputs by at most 0.0000002385. On D, seed 1, test ID 1981 shifted up one pixel with reflection, this reversed a cat/dog logit margin smaller than 0.000001. The prescribed tensor normalization reproduced the archived class. The official calculation uses the frozen protocol.
- The point-in-time sources for the experiment's scope, method and dataset attribution are the authors' publications and dataset instructions. [Azulay and Weiss](https://jmlr.org/papers/v20/19-519.html) · [Zhang, especially §3.1 for circular CIFAR-10 shifts](https://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf) · [CIFAR-10 data and citation instructions](https://www.cs.toronto.edu/~kriz/cifar.html)

| Primary comparison, D minus B | Flip rate (pp) | Clean accuracy (pp) | Shifted accuracy (pp) |
|---|---:|---:|---:|
| Seed 0 | −5.2775 | −0.7000 | −0.6150 |
| Seed 1 | −5.9500 | +0.5600 | +0.5775 |
| Seed 2 | −5.6950 | +0.2500 | +0.3725 |
| Mean | −5.6408 | +0.0367 | +0.1117 |

All three seeds have fewer flips, and both mean accuracy differences lie within the allowed loss of one point. With reflection at four pixels, the mean flip difference is −0.0333 pp and the seed differences vary in sign. Per-image evidence and local checkpoint locations are listed in the [handoff](HANDOFF.md).

## Remaining course actions

**All three members**

1. Confirm individual and shared final-project contributions. Replace the proposal draft on paper page 5 with the confirmed account.
2. Agree on availability and speaking order, rehearse the five-minute presentation and five-minute Q&A, and log in to Gathertown at least 15 minutes before the first presentation.
3. After presenting, record and address the actual TA/instructor feedback. Review the final paper together before Canvas submission.

**Haotian Si, team captain**

1. Verify the Gathertown invitation and edit access, designated graders' repository access, the current poster dimensions, time zone, availability form and assigned slot. The snapshot marks the form TBC and contains conflicting years in Step 4.
2. Submit the team availability form. The snapshot lists December 2, 11:59 pm.
3. Upload the poster PDF to Gathertown first, then to Canvas. The snapshot lists December 8, 5 pm and presentation day December 9. Use the latest course announcement for the binding schedule.
4. Coordinate final report submission to Canvas after contribution and feedback updates. The snapshot lists December 15, 11:59 pm. Verify the submission receipt.
