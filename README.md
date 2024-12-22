# PolypeptideDesiger (PPD)

We introduced PolypeptideDesigner (PPD) model, a conditional text generation model that utilizes per-residue secondary structure conditions to de novo design long polypeptide sequences up to 250 residues in length. Our PPD model combines diffusion-based models with a lightweight LSTM-attention neural network denoiser. PPD can produce more diverse sequences due to its innovative architecture and efficient denoiser, reflecting an enhanced understanding of protein structures.

## Usage

### Dependency

```
Python 3.8.12
PyTorch 2.0.1
TensorFlow 2.11.0
```

The early stage and final stage parameters of PPD models are hosted on [Google Drive](https://drive.google.com/drive/folders/14uJ5ge8_RsOVMUgwjf2_SdjzXTWNju7y?usp=drive_link).

## Reference 
```bibtex
@article{Liao2024PPD,
  title={De Novo Designing of Large Polypeptide Using a Lightweight LSTM and Attention-based Diffusion Model with Per-residue Secondary Structure Condition},
  author={Liao, Sisheng and Xu, Gang and Wang, Qinghua and Ma, Jianpeng},
  journal={Molecules},
  year={2024},
}
