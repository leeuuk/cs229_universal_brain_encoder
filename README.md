# Universal Brain Encoder (Voxel-Centric Image→fMRI Encoding)

This repository is a **modular reference implementation** of the model described in:

*Beliy et al., “The Wisdom of a Crowd of Brains: A Universal Brain Encoder”* (arXiv:2406.12179).  

## Install

**Conda**

```bash
conda env create -f environment.yml
conda activate ube
```

**venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

## Train (multi-subject)

```bash
python scripts/train_nsd.py \
  --config configs/nsd.yaml \
  --data_root /path/to/NSD-processed
```

- By default, subjects are auto-detected from `data_root/sub-*/`.
- Checkpoints are saved to `checkpoints/`.

## Evaluate

Per-subject evaluation (Pearson correlation per voxel + retrieval Top-1/Top-5 on the test split):

```bash
python scripts/eval_nsd.py \
  --config configs/nsd.yaml \
  --data_root /path/to/NSD-processed \
  --ckpt checkpoints/last.pt \
  --subject 1
```

## V2C and cluster-level causal evaluation

To analyze the causal contribution of functional clusters (as in Brain-IT), you can:

1. **Build V2C mapping** (GMM on UBE voxel embeddings, 128 clusters by default):

```bash
python scripts/build_v2c_gmm.py --ckpt checkpoints/best.pt --out data/v2c/
```

2. **Run standard evaluation** to get `eval_tensors.pt` (if not already done):

```bash
python scripts/eval_paper.py --ckpt checkpoints/best.pt --out_dir results/
```

3. **Run cluster ablation**: for each cluster, mask out its voxels and recompute retrieval accuracy:

```bash
python scripts/eval_v2c_ablation.py --eval_dir results/ --v2c_dir data/v2c/ --out_dir results/
```

4. **Plot results** (sorted importance, subject×cluster heatmap, cumulative effect):

```bash
python scripts/plot_v2c_ablation.py --results results/v2c_ablation_results.pt --out_dir results/vis_v2c
```

The drop in Top-1 retrieval when a cluster is removed indicates that cluster’s importance for stimulus-specific encoding. See `ube/v2c.py` for loading V2C and masking helpers.


## Repository layout

- `ube/data/` – datasets + transforms
- `ube/models/` – DINO feature extractor + LoRA + cross-attention + full encoder
- `ube/losses.py` – paper loss
- `ube/metrics.py` – correlation + retrieval
- `ube/v2c.py` – V2C load/ablation mask utilities for cluster-level causal eval
- `scripts/` – training / eval / V2C build & ablation entry points
- `configs/` – YAML configs
