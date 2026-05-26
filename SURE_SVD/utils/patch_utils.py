"""
patch_utils.py
──────────────
Shared utilities for Set12-based patch experiments.

Design
──────
1. Images loaded as float64, normalised to [0, 1].
2. Patch size: 32×32 (good balance of low-rank structure vs. computation).
3. Signal X for each patch = the original patch itself (full rank).
4. "Similar patches" are found by top-K NCC search (no hard threshold),
   so every query always yields K candidates.
5. Monte Carlo replicates = re-sample noise W on the SAME fixed X.
   The similar patches are used to show robustness across image content.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

SET12_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "set12")
IMAGE_NAMES = [f"{i:02d}.png" for i in range(1, 13)]


def load_image(name: str) -> np.ndarray:
    """Load a Set12 image as float64 in [0, 1]."""
    path = os.path.join(SET12_DIR, name)
    return np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def extract_patches(img: np.ndarray, patch_size: int = 32,
                    stride: int = 4) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract all (patch_size × patch_size) patches with given stride."""
    H, W = img.shape
    P = patch_size
    patches, positions = [], []
    for r in range(0, H - P + 1, stride):
        for c in range(0, W - P + 1, stride):
            patches.append(img[r:r + P, c:c + P])
            positions.append((r, c))
    return np.array(patches), positions


def ncc_scores(query: np.ndarray, patches: np.ndarray) -> np.ndarray:
    """Compute NCC between query and every patch in patches."""
    q = query.ravel() - query.mean()
    nq = np.linalg.norm(q)
    if nq < 1e-12:
        return np.zeros(len(patches))
    flat = patches.reshape(len(patches), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    norms = np.maximum(norms, 1e-12)
    return (flat @ q) / (norms * nq)


def find_top_k_similar(query: np.ndarray, patches: np.ndarray,
                        k: int = 50,
                        exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-k most similar patches (by NCC) to query."""
    scores = ncc_scores(query, patches)
    if exclude_idx is not None:
        scores[exclude_idx] = -2.0
    order = np.argsort(scores)[::-1][:k]
    return patches[order], scores[order]


def low_rank_approx(patch: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-r SVD approximation."""
    U, sv, Vt = np.linalg.svd(patch, full_matrices=False)
    sv_trunc = sv.copy()
    sv_trunc[rank:] = 0.0
    return (U * sv_trunc[None, :]) @ Vt, sv


def build_patch_collection(
    image_names: List[str] = None,
    patch_size: int = 32,
    stride: int = 4,
    n_query_per_image: int = 3,
    k_similar: int = 60,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    For each selected query patch from each image:
      1. Find the top-k_similar most similar patches (by NCC) in the same image.
      2. Signal X = the original patch itself (full rank).
    """
    if image_names is None:
        image_names = IMAGE_NAMES
    if rng is None:
        rng = np.random.default_rng(2024)

    collection = []

    for img_name in image_names:
        img = load_image(img_name)
        patches, positions = extract_patches(img, patch_size, stride)
        N = len(patches)

        if N < k_similar + 1:
            if verbose:
                print(f"  {img_name}: only {N} patches, skipping")
            continue

        query_indices = rng.choice(N, size=min(n_query_per_image, N), replace=False)

        for qi in query_indices:
            query = patches[qi]
            query_pos = positions[qi]

            similar, scores = find_top_k_similar(
                query, patches, k=k_similar, exclude_idx=qi
            )

            X = query.copy()
            sv_full = np.linalg.svd(X, full_matrices=False)[1]

            entry = {
                'image': img_name,
                'query': query,
                'query_pos': query_pos,
                'similar': similar,
                'ncc_scores': scores,
                'X': X,
                'sv_full': sv_full,
                'patch_size': patch_size,
            }
            collection.append(entry)

            if verbose:
                print(
                    f"  {img_name} pos={query_pos}: "
                    f"n_similar={len(similar)}, "
                    f"top-NCC={scores[0]:.3f}, "
                    f"median-NCC={np.median(scores):.3f}, "
                    f"top-5 SVs={entry['sv_full'][:5].round(4)}"
                )

    return collection


def tau_from_psnr(patch: np.ndarray, psnr_db: float) -> float:
    """Compute tau corresponding to a given PSNR."""
    return 10 ** (-psnr_db / 20.0)


if __name__ == "__main__":
    print("Building patch collection from Set12 (quick test)...")
    col = build_patch_collection(
        image_names=["01.png", "03.png", "08.png"],
        patch_size=32, stride=4,
        n_query_per_image=2, k_similar=50, verbose=True,
    )
    print(f"\nTotal patch groups: {len(col)}")
