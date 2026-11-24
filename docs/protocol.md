# Frozen experiment and decision rule

We test whether fixed blur still helps when training already includes translations. The primary contrast is D minus B, followed by C minus A. The submitted November 19 proposal sets the outcome definitions, experiment size and decision threshold. The implementation and evaluation were committed before formal training.

| Condition | Both pooling sites | Training translations |
|---|---|---|
| A | 2 × 2 max, stride 2 | None |
| B | 2 × 2 max, stride 2 | Independent uniform integer shifts |
| C | 2 × 2 max, stride 1, then fixed blur at stride 2 | None |
| D | Same as C | Same as B |

The sliding maximum uses valid padding. Its output receives one pixel of edge-exclusive reflection on all sides. The fixed kernel is the outer product of [1, 2, 1] with itself divided by 16, repeated as a depthwise buffer. Output centers sit at sliding-max coordinates 0, 2, 4, …. The dense locations and smoothing follow Richard Zhang (ICML 2019) and the author's antialiased-cnns implementation. The new code implements these operations independently.

Both variants have 83,126 trainable scalars. Layer outputs (channels × height × width) are 6 × 32 × 32, 6 × 16 × 16, 16 × 12 × 12 and 16 × 6 × 6, followed by 576 → 120 → 84 → 10. Both convolutions and the first two dense layers use tanh. The course predecessor used a single input channel and 400 flattened features.

Split official training IDs per class using NumPy PCG64 seed 42. Sort the selected indices after stratification. Estimate population channel means and standard deviations over the 45,000 training images. No flips, crops other than specified translations, or random evaluation transforms. For B/D, each axis independently draws an integer from −2 to +2, including zero, with reflected image boundaries. A positive displacement moves the object right or down. The evaluation mean-fill variant inserts zero after normalization. Network blur padding always stays reflected.

Each seed (0, 1, 2) has its own CPU weight initialization. Order and displacement streams are new NumPy generators keyed by [seed, epoch, 1] and [seed, epoch, 2]. Offsets are assigned by sorted training-subset position before shuffling. The GPU-resident dataset uses no DataLoader workers. Identical orders across conditions and identical B/D offset arrays are verified from all recorded epoch hashes.

Each run makes 704 SGD updates per epoch, retaining the final eight images. Batch size 64, momentum 0.9, weight decay 0.0005, cross-entropy, 60 epochs. Learning rates 0.02, 0.004 and 0.0008 cover epochs 1–25, 26–40 and 41–60. Use the earliest epoch attaining the highest clean validation correct count. CUDA graph replay covers forward/backward on full batches, with optimizer steps outside the graph; graph warmup uses no parameter updates. FP32 throughout with TF32 disabled.

For a fixed checkpoint, compute predictions on the clean image and its left, right, up and down views. The primary test uses a one-pixel displacement and reflection. Flip rate counts disagreement with the clean class across 4N comparisons. Shifted accuracy counts agreement with ground truth across those same views. Clean accuracy uses N original predictions. One training replicate is one seed. A negative D−B flip difference favors blur. Practical benefit requires fewer flips for each of three seeds and mean losses of at most 1 pp for both accuracies. Secondary evaluations cross 1, 2 and 4 pixels with reflect and train-mean fill, using the original validation-selected weights.

Image matching and pilot evaluation do not feed model training. The historical test-selected weights are loaded only by the historical pilot script. The nonpilot sensitivity analysis removes identifiable historical images using image content, never filename indices.
