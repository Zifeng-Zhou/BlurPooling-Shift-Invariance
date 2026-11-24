# Five-minute question period

Choose the question asked and answer in 15–30 seconds. Suggested leads follow the proposed speaking order and can be reassigned.

**Why compare D with B?** Both models receive the same randomly shifted images in the same order. Changing the two pooling stages within a seed estimates the extra benefit of blur after augmentation. C versus A asks the corresponding question without training shifts.

**Is an image a flip if just one direction changes?** Each direction contributes separately. One changed answer contributes one quarter for that image. Across the test set the denominator is 40,000. A changed prediction can even correct a clean error.

**Why blur after sliding maxima?** Taking stride-two maxima discards locations before smoothing. We keep all stride-one maxima, reflect the dense map by one pixel, filter each channel with the binomial kernel, then keep even locations. This changes spatial aggregation while keeping the learned parameter count fixed.

**Did blur make the model more accurate?** The average clean difference was +0.037 pp and the shifted difference was +0.112 pp. Seed 0 lost 0.700 pp on clean images. The three seeds support a consistent reduction in flips here, with mean accuracy within our tolerance. They provide little precision about small accuracy differences.

**What was fixed before opening the test set?** The architecture, split, training schedule, pairing, clean-validation rule, primary statistic and decision threshold. The full test has 10,000 images. We kept the same selected checkpoints for the alternate boundaries and distances.

**Can the historical pilot bias the result?** Its weights were selected using the old course test. We rebuilt it separately and initialized every final model from scratch. Image-content matching identified all 1,000 old test images inside the official test split. Removing them left a 9,000-image subset with a mean flip difference of −5.66 pp.

**Does the training augmentation already solve the problem?** It helps considerably. Standard pooling falls from 28.21% flips in A to 15.68% in B. Blur still reduces B's rate to 10.04%. Without shifts, blur alone gives 15.42% flips, but its mean clean accuracy is below B's.

**What happens beyond the training range?** Four pixels exceeds the two-pixel training range. On reflected borders the D−B flip difference is +0.605, −0.490 and −0.215 pp across seeds. The mean is close to zero. Two stride-two stages have a cumulative stride of four, so phase alignment is one plausible explanation. Lost content and movement relative to the dense head are competing explanations.

**Why test two border rules?** A shift exposes a strip at one edge and crops a strip at the other. Reflection copies interior structure. Mean filling inserts a uniform channel-wise training mean, equivalent to zero after normalization. One-pixel gains persisted with both rules. The internal blur padding did not change.

**Why only three seeds?** The proposal specified three matched repetitions. We show every pair and the sample SD across seeds. Repeated views share both image content and model parameters. Hundreds of thousands of predictions cannot substitute for independently trained models.

**Could a constant predictor pass?** It could have zero flips, but a ten-class constant predictor would have only 10% accuracy on balanced CIFAR-10. The two accuracy tolerances protect the comparison from that kind of shortcut.

**What did you verify when implementing the method?** Shapes, backward gradients, fixed buffers, statistic isolation, known shifts, metric arithmetic and earliest-epoch ties. CUDA graph versus ordinary SGD and interrupted versus uninterrupted training also produced exactly matching weights in small tests on this machine.

**What would you test next?** We would enlarge the canvas to reduce content loss and compare shifts that preserve or change the cumulative sampling phase. Additional seeds could better quantify the small accuracy differences. Neither extension is part of the reported evidence.

**Which member completed each task?** Haotian Si is the captain and Zifeng Zhou authored the earlier assignment. Use the team's confirmed individual contributions when answering. The proposal lists planned roles and the experiment log describes automated work.
