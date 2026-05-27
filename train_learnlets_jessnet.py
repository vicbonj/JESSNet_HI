"""
train_learnlets_jessnet.py — Train Learnlets for JESSNet on the sphere
=======================================================================

Differences from train_learnlets_sia.py:
  1. Patches are extracted from the SPHERICAL wavelet decomposition
     (wt_trans_ducc on HEALPix maps) rather than from the full maps.
     The learnlet receives a 5-scale starlet cube as input.
  2. The dataset is built from HI simulations (CoLoRe) at the correct
     nside, following the same pipeline as JESSNet at inference.
  3. Two Learnlet variants can be trained:
       --thresh percentile  : standard learnlet (no learned k_scales)
       --thresh ksigma      : learnlet with MiniNet k_scales (default)

Usage
-----
    python train_learnlets_jessnet.py \\
        --data /nas/HI21cm/sim_CoLoRe.hd5 \\
        --nside 256 \\
        --nscales 5 \\
        --nside_cut 16 \\
        --thresh ksigma \\
        --epochs 1000 \\
        --output /nas/HI21cm/learnlet_weights/jessnet_learnlet.pth
"""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.model_selection import train_test_split
import healpytools as hpyt
from healpix_manip import from_healpix_to_maps_new_numba
import sys
sys.path.append('/home/bonjean/learnlet/')
from learnlet_sphere import Learnlet


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SphereWaveletDataset(data.Dataset):
    """Patches of spherical wavelet coefficients with random noise injection.

    Each item is a tuple (x_noisy, y_clean, sigma) where:
        x_noisy  : (nscales, H, W) float32  noisy wavelet patch
        y_clean  : (nscales, H, W) float32  clean wavelet patch
        sigma    : ()               float32  noise std used
    """

    def __init__(self, patches_clean: np.ndarray, sigma_max: float = 0.5):
        """
        Parameters
        ----------
        patches_clean : np.ndarray, shape (N, nscales, H, W)
        sigma_max : float
            Maximum noise std (uniform draw in [0, sigma_max]).
        """
        self.patches = torch.from_numpy(patches_clean.astype(np.float32))
        self.sigma_max = sigma_max

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx):
        y = self.patches[idx]                            # (nscales, H, W)
        sigma = torch.rand(1) * self.sigma_max           # scalar
        x = y + torch.randn_like(y) * sigma
        return x, y, sigma.squeeze()


# ---------------------------------------------------------------------------
# Helper: build wavelet patches from HEALPix maps
# ---------------------------------------------------------------------------

def build_wavelet_patches(
    maps: np.ndarray,
    nscales: int,
    nside_cut: int,
) -> np.ndarray:
    """Extract per-patch spherical wavelet coefficients.

    Parameters
    ----------
    maps : np.ndarray, shape (N, p)
        HEALPix maps (one per foreground realisation).
    nscales : int
        Number of wavelet scales (detail scales = nscales-1 + 1 coarse).
    nside_cut : int
        HEALPix nside for patch cutting.

    Returns
    -------
    patches : np.ndarray, shape (N_patches, nscales, H, W)
        Wavelet patches ready for learnlet training.
    """
    all_patches = []

    for m_idx in range(len(maps)):
        x = maps[m_idx:m_idx+1]           # (1, p)
        # Spherical wavelet decomposition
        Xwt = np.rollaxis(
            hpyt.wt_trans_ducc(x, nscales=nscales - 1, nthreads=4), 2, 1
        )  # (1, nscales, p)

        # Reshape to (nscales, p) for patch extraction
        Xwt_flat = Xwt[0]                 # (nscales, p)

        # Extract patches for each scale, then stack
        patches_per_scale = from_healpix_to_maps_new_numba(
            Xwt_flat, nside_cut=nside_cut
        )
        # patches_per_scale : (nscales, n_patches, H, W)
        # Rearrange to (n_patches, nscales, H, W)
        patches_stacked = np.moveaxis(patches_per_scale, 0, 1)
        all_patches.append(patches_stacked)

    return np.concatenate(all_patches, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # ---- Load and preprocess simulations ----
    print(f'Loading data from {args.data} ...')
    with h5py.File(args.data, 'r') as f:
        keys = list(f.keys())
        print(f'  Available keys: {keys}')

        # Load a subset of foreground components
        components = []
        component_keys = [k for k in keys if k != 'HI_signal']  # skip HI
        for key in component_keys[:args.n_components]:
            arr = np.array(f[key])
            # Take a slice if too many realisations
            arr = arr[args.slice_start:args.slice_end]
            print(f'  {key}: {arr.shape}')
            components.append(arr)

        if not components:
            raise ValueError('No foreground component found in the HDF5 file.')

    # Normalise each realisation to unit std
    maps_list = []
    for comp_arr in components:
        for real in comp_arr:
            m = real.copy().astype(np.float32)
            std = np.std(m)
            if std > 0:
                m /= std
            maps_list.append(m)

    maps = np.stack(maps_list, axis=0)   # (N_total, p)
    print(f'Total training maps: {maps.shape[0]}, nside={hpyt.npix2nside(maps.shape[1])}')

    # ---- Build spherical wavelet patches ----
    print('Building spherical wavelet patches ...')
    patches = build_wavelet_patches(maps, args.nscales, args.nside_cut)
    print(f'Patches shape: {patches.shape}')   # (N_patches, nscales, H, W)

    # ---- Train / validation split ----
    patches_train, patches_val = train_test_split(
        patches, test_size=0.1, random_state=42
    )
    print(f'Train: {len(patches_train)}  Val: {len(patches_val)}')

    dataset_train = SphereWaveletDataset(patches_train, sigma_max=args.sigma_max)
    dataset_val   = SphereWaveletDataset(patches_val,   sigma_max=args.sigma_max)

    loader_train = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    loader_val = data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ---- Model ----
    print('Initialising Learnlet ...')
    learnlet = Learnlet(pretrained=False, n_scales=args.nscales).to(device)
    optimizer = torch.optim.AdamW(learnlet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    log2 = torch.log(torch.tensor(2.0)).to(device)

    # ---- Training ----
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        learnlet.train()
        train_loss = 0.0

        for x_noisy, y_clean, sigma in loader_train:
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)
            sigma   = sigma.to(device)

            optimizer.zero_grad()
            # learnlet expects (B, nscales, H, W) and sigma (B,) or (B,1)
            y_pred = learnlet(x_noisy, sigma)
            diff   = y_pred - y_clean
            loss   = (nn.functional.softplus(2 * diff) - diff - log2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learnlet.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader_train)

        learnlet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_noisy, y_clean, sigma in loader_val:
                x_noisy = x_noisy.to(device)
                y_clean = y_clean.to(device)
                sigma   = sigma.to(device)
                y_pred  = learnlet(x_noisy, sigma)
                diff    = y_pred - y_clean
                val_loss += (
                    nn.functional.softplus(2 * diff) - diff - log2
                ).mean().item()
        val_loss /= len(loader_val)

        scheduler.step()

        print(
            f'Epoch {epoch:4d}/{args.epochs}  '
            f'train={train_loss:.4f}  val={val_loss:.4f}  '
            f'lr={scheduler.get_last_lr()[0]:.2e}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(learnlet.state_dict(), args.output)
            print(f'  → Saved best model to {args.output}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}.')
                break

    print(f'Training finished. Best val loss: {best_val_loss:.4f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train JESSNet Learnlet on spherical wavelet patches.'
    )
    parser.add_argument('--data',    type=str,   required=True,
                        help='Path to HDF5 simulation file')
    parser.add_argument('--output',  type=str,   required=True,
                        help='Output path for best model weights (.pth)')
    parser.add_argument('--nside',   type=int,   default=256,
                        help='HEALPix nside of input maps (default: 256)')
    parser.add_argument('--nscales', type=int,   default=5,
                        help='Number of wavelet scales (default: 5)')
    parser.add_argument('--nside_cut', type=int, default=16,
                        help='nside for patch cutting (default: 16)')
    parser.add_argument('--thresh',  type=str,   default='ksigma',
                        choices=['percentile', 'ksigma'],
                        help='Thresholding strategy (default: ksigma)')
    parser.add_argument('--n_components', type=int, default=3,
                        help='Number of foreground component types to load '
                             '(default: 3)')
    parser.add_argument('--slice_start', type=int, default=150,
                        help='Start index for HDF5 slice (default: 150)')
    parser.add_argument('--slice_end',   type=int, default=250,
                        help='End index for HDF5 slice (default: 250)')
    parser.add_argument('--sigma_max', type=float, default=0.5,
                        help='Max noise std for training (default: 0.5)')
    parser.add_argument('--epochs',  type=int,   default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--patience', type=int,  default=20,
                        help='Early stopping patience (default: 20)')

    args = parser.parse_args()
    train(args)
