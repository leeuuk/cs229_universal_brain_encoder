"""Dataset for the Algonauts 2023 Challenge data (NSD).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_image_transform_pil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_subjects(root: str | Path) -> List[int]:
    """Auto-detect subject folders (subj01, subj02, …) under root/train_data/."""
    train_dir = Path(root) / "train_data"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train_data directory not found: {train_dir}")
    subjects: List[int] = []
    for p in train_dir.iterdir():
        m = re.match(r"subj(\d+)", p.name)
        if m and p.is_dir():
            subjects.append(int(m.group(1)))
    return sorted(subjects)


def parse_nsd_id(path: Path) -> int:
    """Extract the NSD 73k image ID from a filename like train-0001_nsd-00013.png."""
    m = re.search(r"nsd-(\d+)", path.name)
    if m is None:
        raise ValueError(f"Cannot parse NSD id from filename: {path.name}")
    return int(m.group(1))


def load_shared_ids(nsd_expdesign_path: str | Path) -> Set[int]:
    """Load the 1,000 shared image IDs (0-indexed 73k IDs) from nsd_expdesign.mat.

    Usage::

        shared = load_shared_ids("path/to/nsd_expdesign.mat")
        dataset = AlgonautsDataset(..., shared_nsd_ids=shared)
    """
    from scipy.io import loadmat
    mat = loadmat(str(nsd_expdesign_path))
    # sharedix is 1-indexed → convert to 0-indexed
    shared_ids = np.asarray(mat["sharedix"]).ravel() - 1
    return set(shared_ids.tolist())


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlgonautsSample:
    image: torch.Tensor     # (3, H, W)  float32, ImageNet-normalised
    fmri: torch.Tensor      # (V,) float32
    subject_id: int
    nsd_id: int             # NSD 73k image ID (0-indexed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Internal key for fMRI / image-path storage: (subject_id, source)
_SourceKey = Tuple[int, str]   # e.g. (1, "train") or (3, "test")


class AlgonautsDataset(Dataset):
    """Algonauts 2023 NSD dataset.

    fMRI is loaded via memory-mapped numpy arrays so that only the accessed
    rows are paged into RAM, keeping multi-subject training feasible.

    Parameters
    ----------
    root : path to algonauts root (containing ``train_data/`` and ``test_data/``)
    split : ``"train"`` | ``"val"`` | ``"shared"``
    subjects : subject ids (1-8).  ``None`` → auto-detect.
    image_size : resize & crop target for DINOv2.
    shared_nsd_ids : set of NSD 73k image IDs (0-indexed) for the 1,000
        shared images.  Use :func:`load_shared_ids` to obtain them.
        When provided, train/val exclude these IDs and the ``"shared"``
        split returns *only* these IDs.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        subjects: Optional[List[int]] = None,
        image_size: int = 224,
        shared_nsd_ids: Optional[Set[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = build_image_transform_pil(image_size)
        self._shared_ids: Set[int] = shared_nsd_ids or set()

        if subjects is None:
            subjects = discover_subjects(self.root)
            if not subjects:
                raise FileNotFoundError(f"No subject folders found under {self.root}")
        self.subjects = sorted(subjects)

        # Per-(subject, source) storage
        self._lh_fmri: Dict[_SourceKey, np.ndarray] = {}
        self._rh_fmri: Dict[_SourceKey, np.ndarray] = {}
        self._img_paths: Dict[_SourceKey, List[Path]] = {}
        self._subject_voxel_sizes: Dict[int, int] = {}

        # Flat index: [(subject_id, source, local_image_idx), ...]
        self._indices: List[Tuple[int, str, int]] = []

        for subj in self.subjects:
            if split == "train":
                self._build_split_train(subj)
            elif split == "val":
                self._build_split_val(subj)
            elif split == "shared":
                self._build_split_shared(subj)
            else:
                raise ValueError(
                    f"Unknown split: {split!r}  (expected train|val|shared)"
                )

        if not self._indices:
            raise FileNotFoundError(
                f"No data found for split={split!r}, subjects={self.subjects}"
            )

        n_vox = list(self._subject_voxel_sizes.values())
        print(f"AlgonautsDataset  split={split}  subjects={self.subjects}  "
              f"samples={len(self._indices)}  voxels/subj={n_vox}")

    # ------------------------------------------------------------------
    # Source loader (shared by all splits)
    # ------------------------------------------------------------------

    def _load_source(
        self, subj: int, source: str,
    ) -> Tuple[List[Path], int]:
        """Load fMRI (mmap) and image paths for a given *source*.

        source = "train" → train_data/subjXX/training_split/{training_fmri, training_images}
        source = "test"  → test_data/subjXX/test_split/{test_fmri, test_images}

        Returns (img_files, n_images).
        """
        key: _SourceKey = (subj, source)
        if key in self._lh_fmri:
            # Already loaded (e.g. shared split loads both sources)
            return self._img_paths[key], self._lh_fmri[key].shape[0]

        subj_name = f"subj{subj:02d}"

        if source == "train":
            subj_dir = self.root / "train_data" / subj_name
            fmri_dir = subj_dir / "training_split" / "training_fmri"
            img_dir = subj_dir / "training_split" / "training_images"
            lh_name, rh_name = "lh_training_fmri.npy", "rh_training_fmri.npy"
            img_glob = "train-*.png"
        elif source == "test":
            subj_dir = self.root / "test_data" / subj_name
            fmri_dir = subj_dir / "test_split" / "test_fmri"
            img_dir = subj_dir / "test_split" / "test_images"
            lh_name, rh_name = "lh_test_fmri.npy", "rh_test_fmri.npy"
            img_glob = "test-*.png"
        else:
            raise ValueError(f"Unknown source: {source!r}")

        lh_path = fmri_dir / lh_name
        rh_path = fmri_dir / rh_name
        if not lh_path.exists() or not rh_path.exists():
            raise FileNotFoundError(f"Missing fMRI for {subj_name}: {fmri_dir}")

        lh = np.load(str(lh_path), mmap_mode="r")
        rh = np.load(str(rh_path), mmap_mode="r")
        n_images = lh.shape[0]

        img_files = sorted(img_dir.glob(img_glob))
        if len(img_files) != n_images:
            raise ValueError(
                f"{subj_name}/{source}: {len(img_files)} images vs "
                f"{n_images} fMRI rows"
            )

        self._lh_fmri[key] = lh
        self._rh_fmri[key] = rh
        self._img_paths[key] = img_files
        self._subject_voxel_sizes[subj] = lh.shape[1] + rh.shape[1]

        return img_files, n_images

    # ------------------------------------------------------------------
    # Split builders
    # ------------------------------------------------------------------

    def _build_split_train(self, subj: int) -> None:
        """training_split (train_data/), shared excluded."""
        img_files, n = self._load_source(subj, "train")
        n_excl = 0
        for i in range(n):
            nsd_id = parse_nsd_id(img_files[i])
            if self._shared_ids and nsd_id in self._shared_ids:
                n_excl += 1
                continue
            self._indices.append((subj, "train", i))
        n_kept = n - n_excl
        print(f"  subj{subj:02d} train: {n_kept} kept" + (f", {n_excl} shared excluded" if n_excl else ""))

    def _build_split_val(self, subj: int) -> None:
        """test_split (test_data/, with fMRI), shared excluded."""
        img_files, n = self._load_source(subj, "test")
        n_excl = 0
        for i in range(n):
            nsd_id = parse_nsd_id(img_files[i])
            if self._shared_ids and nsd_id in self._shared_ids:
                n_excl += 1
                continue
            self._indices.append((subj, "test", i))
        n_kept = n - n_excl
        print(f"  subj{subj:02d} val: {n_kept} kept" + (f", {n_excl} shared excluded" if n_excl else ""))

    def _build_split_shared(self, subj: int) -> None:
        """Shared 1000 images from both training_split and test_split."""
        if not self._shared_ids:
            raise ValueError(
                "split='shared' requires shared_nsd_ids. "
                "Use load_shared_ids() to obtain them."
            )

        n_found = 0

        # From training_split (train_data/)
        img_files_train, n_train = self._load_source(subj, "train")
        for i in range(n_train):
            if parse_nsd_id(img_files_train[i]) in self._shared_ids:
                self._indices.append((subj, "train", i))
                n_found += 1

        # From test_split (test_data/ — a few shared images may land here)
        subj_name = f"subj{subj:02d}"
        test_fmri_dir = (
            self.root / "test_data" / subj_name / "test_split" / "test_fmri"
        )
        if (test_fmri_dir / "lh_test_fmri.npy").exists():
            img_files_test, n_test = self._load_source(subj, "test")
            for i in range(n_test):
                if parse_nsd_id(img_files_test[i]) in self._shared_ids:
                    self._indices.append((subj, "test", i))
                    n_found += 1

        print(f"  subj{subj:02d} shared: {n_found} images found")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def subject_voxel_sizes(self) -> Dict[int, int]:
        return dict(self._subject_voxel_sizes)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> AlgonautsSample:
        subj, source, local_idx = self._indices[idx]
        key: _SourceKey = (subj, source)

        # --- image ---
        img_path = self._img_paths[key][local_idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        nsd_id = parse_nsd_id(img_path)

        # --- fMRI ---
        lh = self._lh_fmri[key][local_idx]   # (V_lh,) mmap slice
        rh = self._rh_fmri[key][local_idx]   # (V_rh,)
        fmri = torch.from_numpy(
            np.concatenate([lh, rh]).astype(np.float32, copy=False)
        )

        return AlgonautsSample(
            image=image,
            fmri=fmri,
            subject_id=subj,
            nsd_id=nsd_id,
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_algonauts(batch: List[AlgonautsSample]) -> Dict[str, Any]:
    """Stack images; keep variable-length fMRI as a list."""
    return {
        "images": torch.stack([s.image for s in batch], dim=0),
        "fmri": [s.fmri for s in batch],
        "subject_ids": [s.subject_id for s in batch],
        "nsd_ids": [s.nsd_id for s in batch],
    }
