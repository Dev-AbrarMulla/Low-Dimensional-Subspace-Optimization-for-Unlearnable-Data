# Low-Dimensional Subspace Optimization for Unlearnable Data

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A project for **CS F320 Foundations of Data Science** at BITS Pilani Dubai.

> **In one line:** class-wise unlearnable-data stamps restricted to *k* PCA or random directions can protect CIFAR-10 images as well as full-space stamps while using about 30 times fewer numbers, as long as the directions are the fine-grain (low-variance) ones.

**Keywords:** unlearnable examples, availability poisoning, error-minimizing noise, shortcut learning, principal component analysis, low-dimensional subspace, data protection, CIFAR-10, ResNet-18, PyTorch.

**Unlearnable data** protects images from being used for training. A small, invisible perturbation (a *stamp*) is added to every training image, so a model trained on the stamped images learns the stamp instead of the content and then fails on clean images. Existing class-wise methods let the stamp change all 3,072 values of a CIFAR-10 image freely. This project asks **how much of that freedom is actually needed**.

We restrict each class stamp to a *k*-dimensional subspace, built from PCA directions (top, middle or bottom of the ranking) or from random directions, optimise it with error-minimizing (min-min) noise under an L∞ budget of 8/255, and measure how well it protects over 76 training runs.

<p align="center"><img src="figures/block_diagram.png" alt="Block diagram of the solution" width="90%"></p>

## Key findings

ResNet-18 trained from scratch for 15 epochs on stamped CIFAR-10, then scored on the 10,000 **clean** test images (lower accuracy means stronger protection; chance is 10%).

| Stamp | Mean clean test accuracy | Protected runs (< 30%) |
|---|---|---|
| No stamp (baseline) | 92.5% | n/a |
| Full space, standard method (5 runs) | 12.7% | 5 of 5 |
| **PCA bottom, k = 100 (5 runs)** | **10.0%** | **5 of 5** |
| Random, k = 100 (5 runs) | 14.7% | 5 of 5 |
| PCA top, k = 100 (5 runs) | 30.7% | 3 of 5 |

- **The stamp does not need all 3,072 dimensions.** Bottom-PCA stamps with k ≥ 100 gave exactly 10.0% (chance) in all 8 runs, matching the standard full-space method with about 30 times fewer numbers (1,000 instead of 30,720).
- **The kind of direction matters more than how many.** Fine-grain (bottom) directions protect fully, smooth (top) directions only partly, even with five times as many of them.
- **A small k is a gamble for every kind of direction.** Runs end near 10% or near 92%, nothing in between, and protection can appear late in training (epochs 9 to 13).
- **A frequency analysis is consistent with fine-grain stamps working best**, but it is not a causal test.
- **Cheaper to describe, not faster to generate**, because training the surrogate model dominates the time.

<p align="center">
  <img src="figures/fig_headline.png" alt="Mean clean test accuracy by stamp type" width="48%">
  <img src="figures/fig_success.png" alt="Share of protected runs by k" width="48%">
</p>

## Background and theory

**Unlearnable data as a bi-level problem.** The owner of a dataset $\{(x_i, y_i)\}$ wants a perturbation $\delta$ that makes training on $x_i + \delta$ useless for the clean distribution. Following Huang et al. (2021), this is written as a min-min problem, where the noise and the model both *minimise* the training loss:

$$\min_{\theta} \; \min_{\delta:\ \|\delta_c\|_\infty \le \epsilon} \; \mathbb{E}_{(x,y)} \Big[ \mathcal{L}\big(f_\theta(x + \delta_{y}),\, y\big) \Big]$$

Because the noise already makes the loss small, the network has little reason to learn the real image content. It latches onto the stamp, a *shortcut feature* that predicts the label, and then fails on clean images. This is the opposite of an adversarial example, which *maximises* the loss at test time. In the **class-wise** setting used here, every image of class *c* receives the same stamp $\delta_c$, so the stamp is an easy, linearly separable signal for the label.

**Why restrict the stamp to a subspace.** A CIFAR-10 image has $32 \times 32 \times 3 = 3{,}072$ values, so the standard class-wise stamp has $10 \times 3{,}072 = 30{,}720$ free numbers. If the stamp is forced to lie in a *k*-dimensional subspace spanned by the columns of an orthonormal matrix $U_k \in \mathbb{R}^{3072 \times k}$, only $10k$ numbers remain (1,000 for $k = 100$). The question is whether protection survives this compression, and which *k* directions to keep.

**Choosing the directions with PCA.** Principal component analysis takes the eigenvectors of the training-set covariance matrix, ordered by the variance they explain. Natural images have most of their energy at low spatial frequencies (a power spectrum that falls roughly as $1/f^{\alpha}$), so:

- **Top components** are smooth, large-scale patterns (colour gradients, blobs). They carry most of the image variance, and a stamp built from them overlaps with the content the model must learn.
- **Bottom components** are fine-grain, high-frequency patterns with very little variance in real images. A stamp built from them barely overlaps with natural image content.
- **Random orthonormal directions** are a control: they cover all frequencies evenly and ignore the data.

One plausible reading of the results is that stamps in low-variance directions are the easiest shortcut for the network to pick up, since nothing in the clean data competes for those directions. The frequency analysis in the demo notebook is consistent with this, but it is a correlation, not a causal test.

**Keeping the budget exactly.** The stamp for class *c* is a weighted mix of the chosen directions, rescaled so its largest entry equals the budget:

$$\delta_c = \epsilon \cdot \frac{U_k a_c}{\max \left| U_k a_c \right|}$$

Clipping $U_k a_c$ to $[-\epsilon, \epsilon]$ would push it out of the subspace. Rescaling keeps it inside, and the L∞ norm is exactly $\epsilon = 8/255 \approx 0.0314$ for every run (see the `linf` column in `results/results.csv`). The weights $a_c$ are learned, so the optimisation is over $10k$ numbers rather than $30{,}720$.

**How protection is measured.** A fresh ResNet-18 is trained from scratch on the stamped training set and scored on the 10,000 *clean* test images. Accuracy near 10% is chance, so the stamp fully worked. Accuracy near 92.5%, the unstamped baseline, means the model ignored the stamp. A run counts as *protected* below 30%. Runs end close to one of these two extremes, which is why the results report the share of protected runs and not only the mean.

## Method in brief

Each class *c* gets *k* learnable weights *a<sub>c</sub>*, and its stamp is a rescaled mix of the chosen directions *U<sub>k</sub>*:

$$\delta_c = \epsilon \cdot \frac{U_k a_c}{\max \left| U_k a_c \right|}, \qquad \epsilon = 8/255$$

The mixture is rescaled instead of clipped, so the stamp stays inside the subspace. A surrogate ResNet-18 and the weights *a<sub>c</sub>* take turns improving (50 SGD steps for the surrogate, then 20 Adam steps for the weights) until the surrogate reaches 99% accuracy on stamped images or 40 rounds are done. A fresh ResNet-18 is then trained on the stamped data and scored on clean test images.

## Repository contents

| Path | What it is |
|---|---|
| `notebooks/FDS_Training_Project.ipynb` | Experiment code: PCA, stamp optimisation, training and evaluation, and the queues of runs |
| `notebooks/FDS_Demo.ipynb` | Demo notebook: clean vs. protected images, a live check of the protected model, stamps and directions, frequency analysis, result plots |
| `results/results.csv` | One row per run: method, k, region, seed, clean test accuracy, L∞ norm, seconds |
| `figures/` | Result figures and the block diagram |

## Running the notebooks

Both notebooks were written for **Google Colab with a GPU** and keep their files in a Google Drive folder named `FDS_ULD_PCA` (cached CIFAR-10, saved stamps, model weights and `results.csv`). The saved stamps and weights are not included here because of their size. The notebooks show their saved outputs when opened on GitHub.

To reproduce the results: run the setup cells of the training notebook first, then the queue cells. The queue cells skip runs already recorded in `results.csv`, so they are safe to re-run after a disconnect. One run takes about 3 minutes on an L4 GPU. Python dependencies are in `requirements.txt`.

## Limitations

One dataset (CIFAR-10), one model (ResNet-18), class-wise noise only, one budget (8/255) and a fixed 15-epoch recipe, with 3 to 5 runs per setting. Defences such as blurring or JPEG compression were not tested, and the comparison with recent unlearnable-data methods is qualitative.

## Related work

- H. Huang et al., *Unlearnable examples: Making personal data unexploitable*, ICLR 2021.
- J. Li et al., *A survey on unlearnable data*, arXiv:2503.23536, 2025.
- M. Li et al., *Detecting and corrupting convolution-based unlearnable examples*, arXiv:2311.18403, 2023.
- Y. Zhe and J. Sakuma, *Beyond full poisoning: Effective availability attacks with partial perturbation*, arXiv:2407.02437, 2024.
- J. Cai et al., *FUSE: Full-spectrum unlearnable examples via spectral equalization*, ICML 2026.

## License

Released under the [MIT License](LICENSE).
