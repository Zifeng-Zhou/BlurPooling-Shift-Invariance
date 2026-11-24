# Five-minute poster presentation

Suggested speaking order only. Swap speakers freely. At about 120 words per minute, allow a short pause to point to each figure. Bracketed cues and headings are silent.

## 0:00–1:30 · Zifeng Zhou · The question

[Point to the truck.]

Look at these two images. In the second one, the truck has moved one pixel to the right. It is still the same truck. The bright regions in the third panel show where the pixel values changed most. Those changes are concentrated along edges. If a strong edge moves across the boundary of a pooling window, the classifier can keep a different response and eventually predict a different class.

We first saw this in an earlier assignment. We reloaded the saved grayscale model and moved each of a thousand course images one pixel in four directions. The predicted class changed 816 times in four thousand comparisons, just over twenty percent. That checkpoint had been selected on the course test set, so we used it to motivate the question and started the final experiment with fresh weights.

Our question is whether a fixed blur still helps once the network has already learned from shifted images. We care about accuracy as well as stability. A model that confidently gives the same wrong label after every shift would be very stable and of little practical use.

## 1:30–3:05 · Hongbo Yu · The comparison

[Point across the two columns of the experiment table.]

We used standard color CIFAR-10. From the official training images we set aside five hundred per class for clean validation. That left forty-five thousand images for learning, and the official ten thousand test images for evaluation.

The table has two choices. Across the columns, we either keep the training images still or move them randomly by up to two pixels in each direction. Down the rows, we change how both pooling stages work. Standard pooling keeps a maximum and steps forward two pixels. The blur model takes all the overlapping maxima first, averages nearby responses with a fixed weighted filter, and only then takes every second position. Both models have exactly the same number of trainable parameters.

We trained all four combinations for three seeds. Within each seed, every model started with identical learned weights and saw images in the same order. The two augmented models also received the same random shifts. We picked the earliest checkpoint with the best clean validation accuracy. At test time we compared each clean label prediction with four shifted predictions. Every direction counts once.

## 3:05–5:00 · Haotian Si · Findings and limit

[Point to the paired bars.]

The two bars for each seed are the models that both trained with shifts. The standard model changed its answer on about sixteen percent of the one-pixel views. With blur, that fell to about ten percent. Reading across seeds, the reductions were 5.28, 5.95, and 5.69 percentage points.

Our decision rule required fewer flips in every seed and no more than a one-point loss in either average accuracy. Clean accuracy changed by plus 0.04 points, and shifted accuracy by plus 0.11. The rule passed. That average hides a trade-off in the first seed, where blur lost 0.70 points of clean accuracy. The stability finding is much clearer than any claim about an accuracy gain.

Would the result survive a different way of filling the exposed edge? At one pixel, filling with the training mean gave a similar reduction. Distance made a larger difference. At two pixels the reflected-border advantage fell to 2.82 points. At four pixels the difference was only 0.03 points on average.

So for this small classifier, blur adds measurable stability beyond translation training at one pixel. If the positioning errors you expect are four pixels wide, our experiment gives much less reason to choose it. Thank you.
