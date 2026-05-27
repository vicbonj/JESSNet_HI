"""
jessnet.py — JESSNet: Joint Estimation of Sky Sources Network
=============================================================

HI 21-cm foreground removal algorithm combining:
  - Coarse-scale PCA subtraction
  - SDecGMCA-style joint deconvolution + BSS on detail scales
  - Learnlet thresholding on the sphere (two strategies)

The goal is NOT to recover individual foreground components accurately,
but to produce a clean estimate of B*(A@S) that can be subtracted from
the observed data to reveal the HI signal + noise.

Two thresholding variants are available:

  JESSNet(thresh='percentile')
      Per-scale percentile threshold, linearly decreasing from p_K_max
      to p_K_min over the iterations. Identical in spirit to the strategy
      in sdecgmca_thresh.py.

  JESSNet(thresh='ksigma')
      k_scales * 1.5 * sigma_NMAD, where k_scales are the learned scale
      weights stored inside the Learnlet network (learnlet.k_scales or
      learnlet.mininet). No decreasing schedule — the threshold is always
      noise-adaptive.

Usage
-----
    from jessnet import JESSNet

    jess = JESSNet(
        X, beam_model, n=4,
        weight_path='weights/learnlet_hi.pth',
        thresh='percentile',       # or 'ksigma'
        nscales=5, nbItMin=100,
        c_wu=np.array([1e-2, 1e-3]),
    )
    jess.run()
    RecHI = X - jess.fgd_convolved   # foreground-subtracted residual

Dependencies
------------
numpy, torch, healpytools (hpyt), healpix_manip, learnlet_sphere
"""

from __future__ import annotations

import numpy as np
import torch
import healpytools as hpyt
from healpix_manip import (
    from_healpix_to_maps_new_numba,
    from_maps_to_healpix_new_numba,
)
import sys
sys.path.append('/home/bonjean/learnlet/')
from learnlet_sphere import Learnlet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation noise estimator (1-D or per-row 2-D)."""
    if x.ndim == 1:
        return float(np.median(np.abs(x - np.median(x))) / 0.6735)
    return np.median(np.abs(x - np.median(x, axis=1)[:, None]), axis=1) / 0.6735


def _safe_pinv(A: np.ndarray) -> np.ndarray:
    """Numerically stable pseudo-inverse via SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    floor = np.max(s) * 1e-9
    s = np.where(s < floor, floor, s)
    return (Vt.T * (1.0 / s)) @ U.T


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JESSNet:
    """HI foreground removal via joint deconvolution BSS on the sphere.

    Parameters
    ----------
    X : np.ndarray, shape (m, p)
        Observed frequency maps (m frequencies, p HEALPix pixels).
    Hl : np.ndarray, shape (m, lmax+1)
        Beam transfer functions per frequency.
    n : int
        Number of foreground components to model.
    weight_path : str or list of str
        Path to pre-trained Learnlet weight file(s).
        If a single path, the same network is used for all components.
        If a list of length n, each component gets its own network.
    thresh : str
        Thresholding strategy: ``'percentile'`` or ``'ksigma'``.
    nscales : int
        Number of wavelet scales (detail + 1 coarse). Default: 5.
    nbItMin : int
        Number of iterations. Default: 100.
    nside_cut : int
        HEALPix nside for patch cutting in learnlet thresholding. Default: 16.
    c_wu : float or np.ndarray of shape (2,)
        Tikhonov regularization at warm-up (decreases geometrically if array).
        Default: np.array([1e-2, 1e-3]).
    cwuDec : int
        Number of iterations for c_wu decrease. Default: nbItMin // 2.
    p_K_max : list of float, length n
        Starting percentile per component (percentile thresh only).
        Default: [99.99] * n.
    p_K_min : list of float, length n
        Final percentile per component (percentile thresh only).
        Default: [90.0] * n.
    var_noise : np.ndarray of shape (m,) or None
        Per-frequency noise variance (used for noise propagation to sources).
    galmask : np.ndarray of shape (p,) or None
        Boolean mask (True = valid pixel). Default: all True.
    verb : int
        Verbosity level 0–2. Default: 1.
    """

    def __init__(
        self,
        X: np.ndarray,
        Hl: np.ndarray,
        n: int,
        weight_path: str | list[str],
        thresh: str = 'percentile',
        nscales: int = 5,
        nbItMin: int = 100,
        nside_cut: int = 16,
        c_wu=None,
        cwuDec: int | None = None,
        p_K_max: list | None = None,
        p_K_min: list | None = None,
        var_noise: np.ndarray | None = None,
        galmask: np.ndarray | None = None,
        verb: int = 1,
    ):
        assert thresh in ('percentile', 'ksigma'), \
            "thresh must be 'percentile' or 'ksigma'"

        self.X = X
        self.Hl = Hl
        self.n = n
        self.thresh = thresh
        self.nscales = nscales
        self.nbItMin = nbItMin
        self.nside_cut = nside_cut
        self.c_wu = np.array([1e-2, 1e-3]) if c_wu is None else np.atleast_1d(c_wu)
        self.cwuDec = nbItMin // 2 if cwuDec is None else cwuDec
        self.p_K_max = [99.99] * n if p_K_max is None else list(p_K_max)
        self.p_K_min = [90.0]  * n if p_K_min is None else list(p_K_min)
        self.var_noise = var_noise
        self.verb = verb

        self.m, self.p = X.shape
        self.nside = hpyt.npix2nside(self.p)
        self.lmax = Hl.shape[1] - 1
        self.t = hpyt.getsize(self.lmax)
        self.ls, self.ms = hpyt.getlm(self.lmax)
        self.factors = 2 - (self.ms == 0)
        self.Hlm = np.concatenate([Hl[:, l:] for l in range(self.lmax + 1)], axis=1)

        # Mask
        if galmask is None:
            self.galmask = np.ones(self.p, dtype=bool)
        else:
            self.galmask = galmask.astype(bool)

        # ------------------------------------------------------------------
        # Coarse / detail split (spherical wavelet on the sphere)
        # ------------------------------------------------------------------
        if self.verb:
            print('[JESSNet] Computing spherical wavelet decomposition ...')
        Xwt = np.rollaxis(
            hpyt.wt_trans_ducc(X, nscales=self.nscales - 1, nthreads=30), 2, 1
        )
        self.X_coarse = Xwt[:, -1]   # (m, p)
        self.X_det    = X - self.X_coarse  # (m, p)

        # ------------------------------------------------------------------
        # Coarse scale: PCA initialisation of A on X_coarse
        # (no ILC — we just need a good A for foreground subtraction)
        # ------------------------------------------------------------------
        if self.verb:
            print('[JESSNet] PCA on coarse scale ...')
        Xlm_coarse = hpyt.map2alm_ducc(self.X_coarse, lmax=self.lmax, nthreads=30)
        # Dewhiten with first beam to get a scale-invariant covariance
        Xlm_eq = hpyt.alm_product_ducc(Xlm_coarse, Hl[0] / (Hl + 1e-10))
        R = np.real(Xlm_eq @ Xlm_eq.T.conj())
        _, V = np.linalg.eigh(R)
        self.A = V[:, ::-1][:, :n].copy()
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)

        # ------------------------------------------------------------------
        # Detail scales: alm of X_det (masked for A update)
        # ------------------------------------------------------------------
        self.Xlm = hpyt.map2alm_ducc(self.X_det, lmax=self.lmax, nthreads=30)
        X_det_masked = self.X_det * self.galmask[:, None].T  # broadcast-safe
        X_det_masked = self.X_det.copy()
        for i in range(self.m):
            X_det_masked[i] = self.X_det[i] * self.galmask
        self.Xlm_det = hpyt.map2alm_ducc(X_det_masked, lmax=self.lmax, nthreads=30)

        # Working arrays
        self.S    = np.zeros((n, self.p))
        self.Slm  = np.zeros((n, self.t), dtype=complex)
        self.Slm_det = np.zeros((n, self.t), dtype=complex)

        # ------------------------------------------------------------------
        # Load Learnlet(s)
        # ------------------------------------------------------------------
        if self.verb:
            print('[JESSNet] Loading Learnlet weights ...')
        if isinstance(weight_path, str):
            weight_paths = [weight_path] * n
        else:
            assert len(weight_path) == n, \
                'weight_path must be a string or a list of length n'
            weight_paths = weight_path

        self.learnlets = []
        for wp in weight_paths:
            net = Learnlet(pretrained=False, n_scales=self.nscales).cuda()
            net.load_state_dict(
                torch.load(wp, map_location='cuda', weights_only=True)
            )
            net.eval()
            self.learnlets.append(net)

        # Output attributes (filled after run())
        self.fgd_convolved: np.ndarray | None = None  # B*(A@S)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run JESSNet and store the convolved foreground in self.fgd_convolved."""
        if self.verb:
            print(f'[JESSNet] Running ({self.nbItMin} iterations, '
                  f"thresh='{self.thresh}') ...")
        self._core()
        # Convolve foreground estimate with beam and add coarse
        AS = self.A @ self.S                              # (m, p) detail fg
        AS_coarse = self.A @ (_safe_pinv(self.A) @ self.X_coarse)  # coarse fg
        fg_total = AS + AS_coarse
        self.fgd_convolved = hpyt.convolve(fg_total, self.Hl)
        if self.verb:
            print('[JESSNet] Done.')

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _c_at(self, it: int) -> float:
        """Current Tikhonov regularization coefficient."""
        if np.isscalar(self.c_wu) or len(self.c_wu) == 1:
            return float(self.c_wu)
        c_max, c_min = float(np.max(self.c_wu)), float(np.min(self.c_wu))
        denom = max(self.cwuDec - 1, 1)
        return max(
            c_min,
            c_max * 10 ** (
                (np.log10(c_min) - np.log10(c_max)) * (it - 1) / denom
            ),
        )

    def _core(self) -> None:
        """Main alternating optimisation loop."""
        S_old = np.zeros((self.n, self.p))
        A_old = np.zeros_like(self.A)

        for it in range(1, self.nbItMin + 1):
            c = self._c_at(it)

            # ---- Update S via Tikhonov-regularised least-squares ----
            self._ls_s(c)

            # ---- Learnlet thresholding in wavelet domain ----
            self._threshold_s(it)

            # ---- Update A ----
            self._update_a()

            # ---- Convergence diagnostics ----
            mask = self.galmask
            norm_S = np.linalg.norm(self.S[:, mask])
            delta_S = (
                np.linalg.norm((self.S - S_old)[:, mask]) / norm_S
                if norm_S > 0 else np.inf
            )
            delta_A = np.max(np.abs(1 - np.abs(np.sum(self.A * A_old, axis=0))))
            S_old = self.S.copy()
            A_old = self.A.copy()

            if self.verb >= 1:
                print(f'  it={it:4d}  c={c:.2e}  delta_S={delta_S:.2e}  '
                      f'delta_A={delta_A:.2e}  cond(A)={np.linalg.cond(self.A):.1f}')

    def _ls_s(self, c: float) -> None:
        """Tikhonov-regularised least-squares update of S in the SH domain.

        Uses mixing-matrix-based regularisation (strat=1 of SDecGMCA):
            Ra[ℓ] = A^T diag(Hl[ℓ]²) A + ε I
            Slm[ℓ] = Ra[ℓ]^{-1} A^T diag(Hl[ℓ]) Xlm[ℓ]
        """
        Ra  = np.einsum('il,jl,kl->ijk', self.factors * self.Hlm**2,
                        self.A.T[:, :, None] * np.ones((1, 1, self.t)),
                        self.A.T[:, :, None] * np.ones((1, 1, self.t)))
        # Ra has shape (n, n, t)
        # Compute per-ℓ epsilon for mixing-matrix-based regularisation
        AtA_norm = np.linalg.norm(self.A.T @ self.A)
        # Per-ℓ Ra matrices and inversion
        Slm_new = np.zeros_like(self.Slm)
        for l_idx in range(self.t):
            Ra_l = np.real(
                np.einsum('i,j->ij',
                          (self.Hlm[:, l_idx]**2),
                          np.ones(self.n)) *
                (self.A.T @ self.A)
            )
            # Mixing-matrix-based epsilon
            eig_min = np.linalg.eigvalsh(Ra_l).min()
            eps_l = max(0.0, c - eig_min / (AtA_norm + 1e-30)) * np.eye(self.n)
            Ra_reg = Ra_l + eps_l
            # piA[ℓ] = Ra_reg^{-1} A^T diag(Hl[ℓ])
            try:
                iRa = np.linalg.inv(Ra_reg)
            except np.linalg.LinAlgError:
                iRa = _safe_pinv(Ra_reg)
            piA_l = iRa @ (self.A.T * self.Hlm[:, l_idx])
            Slm_new[:, l_idx] = piA_l @ self.Xlm[:, l_idx]

        self.Slm = Slm_new

    def _threshold_s(self, it: int) -> None:
        """Apply learnlet thresholding to S in the spherical wavelet domain."""
        # Reconstruct S from Slm
        self.S = hpyt.alm2map_ducc(self.Slm, nside=self.nside, nthreads=30)

        # Spherical wavelet decomposition of S
        S_detwt = np.rollaxis(
            hpyt.wt_trans_ducc(self.S, nscales=self.nscales - 1, nthreads=30),
            2, 1,
        )  # shape (n, nscales, p)
        S_detwt[:, -1] *= 0   # zero out coarse scale
        S_detwt[:,  0] *= 0   # zero out finest scale (below beam resolution)

        # Compute thresholds
        if self.thresh == 'percentile':
            threshold_perc = self._compute_percentile_thresholds(S_detwt, it)
        else:
            threshold_perc = self._compute_ksigma_thresholds(S_detwt)

        # Reshape for learnlet forward pass (patches on sphere)
        S_flat = S_detwt.reshape(self.n * self.nscales, self.p)
        patch_S = from_healpix_to_maps_new_numba(S_flat, nside_cut=self.nside_cut)
        n_patches = patch_S.shape[1]
        patch_S = (
            patch_S.reshape(self.n, self.nscales, n_patches,
                            patch_S.shape[2], patch_S.shape[3])
        ).swapaxes(1, 2)
        # patch_S : (n, n_patches, nscales, H, W)

        patch_S_out = np.empty(
            (self.n, n_patches, patch_S.shape[3], patch_S.shape[3]),
            dtype=np.float32,
        )

        patches_torch = torch.from_numpy(patch_S).float()
        batch_size = 400
        with torch.no_grad():
            for i in range(self.n):
                for start in range(0, n_patches, batch_size):
                    batch = patches_torch[i, start:start + batch_size].cuda()
                    thr_i = torch.from_numpy(
                        threshold_perc[i].astype(np.float32)
                    ).cuda()
                    out = self.learnlets[i](
                        batch, sigma=None,
                        no_coarse=False,
                        thresholds_hc=thr_i,
                    )
                    patch_S_out[i, start:start + len(batch)] = (
                        out[:, 0].cpu().numpy()
                    )

        self.S = from_maps_to_healpix_new_numba(patch_S_out)

        # Mask outside valid region
        for i in range(self.n):
            self.S[i] *= self.galmask

        # Update Slm_det for A update
        self.Slm_det = hpyt.map2alm_ducc(self.S, lmax=self.lmax, nthreads=30)

    def _compute_percentile_thresholds(
        self, S_detwt: np.ndarray, it: int
    ) -> np.ndarray:
        """Per-scale percentile thresholds, linearly decreasing over iterations.

        Returns
        -------
        threshold_perc : np.ndarray, shape (n, nscales-1)
            Thresholds[comp][scale] to pass to learnlet as thresholds_hc.
        """
        p_K = np.array([
            self.p_K_max[comp]
            - (self.p_K_max[comp] - self.p_K_min[comp]) * it / self.nbItMin
            for comp in range(self.n)
        ])

        threshold_perc = []
        for j in range(self.nscales - 1):
            row = np.array([
                np.percentile(
                    np.abs(S_detwt[comp, j][self.galmask]), p_K[comp]
                )
                for comp in range(self.n)
            ])
            threshold_perc.append(row)
        # shape (nscales-1, n) → (n, nscales-1)
        return np.array(threshold_perc).T

    def _compute_ksigma_thresholds(
        self, S_detwt: np.ndarray
    ) -> np.ndarray:
        """k_scales * 1.5 * sigma_NMAD thresholds using learned scale weights.

        The Learnlet network stores learned per-scale multipliers accessible
        via ``learnlet.mininet`` (the MiniNet sub-network).  We call it with
        a unit input to retrieve the k_scales vector and multiply by
        1.5 * NMAD noise estimate.

        Returns
        -------
        threshold_perc : np.ndarray, shape (n, nscales-1)
        """
        threshold_perc = []
        k_unit = torch.ones(1, 1, device='cuda')

        for i in range(self.n):
            # Retrieve learned scale weights from MiniNet
            with torch.no_grad():
                k_scales = (
                    self.learnlets[i].mininet(k_unit) * 5.0
                ).cpu().numpy().ravel()   # shape (nscales-1,) or similar
            # k_scales may have fewer entries than nscales-1; broadcast safely
            n_k = min(len(k_scales), self.nscales - 1)

            row = np.zeros(self.nscales - 1)
            for j in range(self.nscales - 1):
                coeffs = S_detwt[i, j][self.galmask]
                sigma_nmad = 1.4826 * np.median(np.abs(coeffs - np.median(coeffs)))
                k_j = float(k_scales[j]) if j < n_k else float(k_scales[-1])
                row[j] = k_j * 1.5 * sigma_nmad
            threshold_perc.append(row)

        return np.array(threshold_perc)  # (n, nscales-1)

    def _update_a(self) -> None:
        """Least-squares update of A in the spherical harmonic domain.

        A = (∑_{ℓm} Hl² Slm Slm†)^{-1} (∑_{ℓm} Hl Xlm_det Slm†)
        """
        Rs = np.real(
            np.einsum(
                'il,jl,kl',
                self.factors * self.Hlm**2,
                self.Slm_det,
                np.conj(self.Slm_det),
            )
        )  # (n, n)

        Ws = np.real(
            np.einsum(
                'ij,kj->ik',
                self.factors * self.Xlm_det * self.Hlm,
                np.conj(self.Slm_det),
            )
        )  # (m, n)

        # Regularised inversion of Rs per source (diagonal blocks)
        # Use SVD-based pseudo-inverse with floor
        U, s, Vt = np.linalg.svd(Rs)
        floor = np.max(s) * 1e-9
        s = np.where(s < floor, floor, s)
        iRs = (Vt.T * (1.0 / s)) @ U.T

        self.A = Ws @ iRs
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)
