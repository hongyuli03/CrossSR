# CrossSR

Official implementation of **CrossSR: Rethinking Cross-dataset Activation for Diffusion-Based Image Super-Resolution via Neuron-Memory-based Provenance Indexing**.

CrossSR is a target-adaptive framework for cross-dataset training data construction in diffusion-based image super-resolution. Its core component is **Neuron-Memory-based Provenance Indexing (NMPI)**, which leverages target-induced neuron-memory priors to evaluate, rank, and select relevant training samples from multiple source datasets.

To further mitigate low-frequency style discrepancies, CrossSR incorporates **Wavelet-based Test-time Alignment (WTA)** as an auxiliary calibration module. WTA is applied only after NMPI-based sample selection and does not participate in candidate scoring, ranking, or training set composition.

The core implementation is provided in [`Cross_SR.py`](./Cross_SR.py).

## CrossSR Dataset

We provide the **target-tailored training datasets** constructed using our proposed **NMPI** method for different target test scenarios.

**Download:** [Baidu Drive](https://pan.baidu.com/s/1ayfCS43X18zNZH5DOwsAlQ?pwd=5k5c)

## Training Frameworks

CrossSR focuses on target-adaptive training data construction rather than designing new super-resolution architectures. To evaluate its effectiveness and cross-model applicability, we conduct experiments using three existing diffusion-based image super-resolution frameworks: **HYPIR, FiDeSR, and LPNSR**.

The corresponding training implementations are provided in this repository and are based on the following original works:

- **HYPIR** — *HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration* (SIGGRAPH 2025).  
  Original repository: [XPixelGroup/HYPIR](https://github.com/XPixelGroup/HYPIR)  
  Training code: [`HYPIR_train/`](./HYPIR_train/)

- **FiDeSR** — *FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution* (CVPR 2026).  
  Original repository: [Ar0Kim/FiDeSR](https://github.com/Ar0Kim/FiDeSR)  
  Training code: [`FiDeSR_train/`](./FiDeSR_train/)

- **LPNSR** — *LPNSR: Learnable Noise Prediction for Diffusion-Based Image Super-Resolution*.  
  Original repository: [Faze-Hsw/LPNSR](https://github.com/Faze-Hsw/LPNSR)  
  Training code: [`LPNSR_train/`](./LPNSR_train/)

In our experiments, these super-resolution models are trained using the target-tailored datasets constructed by CrossSR. The underlying model architectures and training protocols are kept unchanged across different training data configurations.

This allows us to evaluate the effectiveness of NMPI-driven cross-dataset training data construction across different diffusion-based super-resolution frameworks.

## Acknowledgments

We sincerely thank the authors of HYPIR, FiDeSR, and LPNSR for making their valuable research and implementations publicly available.

Our experimental implementations build upon their original codebases. We acknowledge their contributions and refer users to the corresponding repositories for the original methods, implementation details, and licensing information.

The primary methodological contribution of CrossSR lies in **NMPI-driven cross-dataset training data construction**, while WTA serves as an auxiliary module for low-frequency style calibration. The underlying super-resolution architectures are credited to their respective original authors.

## License and Attribution

The original implementations of HYPIR, FiDeSR, and LPNSR remain subject to their respective licenses and copyright notices. Any redistributed or modified third-party components must comply with the applicable license terms.

The source images used to construct the CrossSR training datasets also remain subject to the respective original dataset licenses and usage conditions.

Please refer to the original repositories and dataset providers for their licensing conditions.
