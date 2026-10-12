# LPNSR: Learnable Noise Prediction for Diffusion-Based Image Super-Resolution

[![Paper](https://img.shields.io/badge/arXiv-2603.21045-b31b1b.svg)](https://arxiv.org/abs/2603.21045)

## Visual Results

<div align="left">
  <b>4× Real-world Super-Resolution</b>
  <br><br>
  <table>
    <tr>
      <th>LQ Image</th>
      <th>SR Image</th>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/oldphoto6.png" width="300"></td>
      <td><img src="results/oldphoto6.png" width="300"></td>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/Lincoln.png" width="300"></td>
      <td><img src="results/Lincoln.png" width="300"></td>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/0003.jpg" width="300"></td>
      <td><img src="results/0003.png" width="300"></td>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/0014.jpg" width="300"></td>
      <td><img src="results/0014.png" width="300"></td>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/29.jpg" width="300"></td>
      <td><img src="results/29.png" width="300"></td>
    </tr>
    <tr>
      <td><img src="assets/RealSet80/0030.jpg" width="300"></td>
      <td><img src="results/0030.png" width="300"></td>
    </tr>
  </table>
</div>

## Requirements

- Python 3.10.11, PyTorch 2.9.1+cu128, Xformers 0.0.33.post2
- A suitable conda environment named `lpnsr` can be created and activated with:

```bash
conda create -n lpnsr python=3.10.11
conda activate lpnsr

# Install PyTorch with CUDA support 
#CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt

# Linux/headless environments: remove the GUI OpenCV wheel that may be
# installed transitively by basicsr/facexlib, then restore the headless wheel.
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless

# Install xformers for acceleration
pip install xformers==0.0.33.post2
```

## Pre-trained Models

Download all the required weights from the [Releases](https://github.com/Faze-Hsw/LPNSR/releases) page of this repository, and place them in the `pretrained/` folder (you can change the weights path in `configs/inference.yaml`):

| Model                                                 | Description |
|-------------------------------------------------------|-------------|
| `autoencoder_vq_f4.pth`                               | VQGAN encoder/decoder (4x spatial compression) |
| `resshift_realsrx4_s4_v3.pth`                         | Pre-trained ResShift UNet |
| `noise_predictor_v2.pth`                              | V2 version of the noise predictor (End-to-end supervised training.) |
| `noise_predictor_v3.safetensors`                      | V3 version of the noise predictor (trained on a larger dataset) (End-to-end supervised training.) |
| `residual.safetensors`                                | Residual corrector (Mean-correction training.) |

Note: The baseline `noise_predictor.pth` is the exact model checkpoint used for the experiments presented in the paper. The v2 version computes GAN loss in the latent space, and achieves better generation quality.


## Quick Start

### :rocket: Inference

```bash
python inference.py -i [image folder/image path] -o [output folder]
```

### :test_tube: Testing

```bash
python test.py --lq_folder [lq image folder] --gt_folder [gt image folder] --output_folder [output folder]
```

Computes PSNR (Y), SSIM (Y), LPIPS, DISTS, MUSIQ, MANIQA and CLIPIQA per image, plus dataset-level FID when GT images are provided. Metrics are saved to `metrics.csv` / `metrics.json` together with the SR images in the output folder.

**Note:** If only LQ images are provided (without GT reference images), only no-reference metrics will be computed.

## Training

### :turtle: Preparing Stage

1. Create a folder named `traindata/` and put your training data in the `traindata/` folder (high-resolution images)
2. Download the pre-trained models (see above)
3. Adjust the configuration in `configs/trainer.yaml`

### :dolphin: Begin Training

```bash
python trainer.py
```

Training is iteration-based (`training.iterations`, default 200000 optimizer steps). Validation on `assets/validate_lq` / `assets/validate_gt` (PSNR, SSIM, LPIPS, DISTS, MUSIQ, MANIQA, CLIPIQA) runs automatically every `save_freq` steps.

### :whale: Resume from Interruption

```bash
python trainer.py --resume experiments/noise_predictor/checkpoints/training_state_stepXXXXX.pth
```

## Acknowledgement

This project is based on:
- [ResShift](https://github.com/zsyOAOA/ResShift) - Efficient diffusion model for image SR
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - Basic super-resolution toolbox
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Degradation simulation

Thanks for these awesome works.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions, please feel free to open an issue or contact the maintainer via `frozen2001@hust.edu.cn`.
