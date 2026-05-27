"""
run_jessnet.py — JESSNet foreground removal pipeline for HI 21-cm
=================================================================

Drop-in replacement for run_SDecGMCA_galmask_newbrightmask.py.
Reads the same HDF5 format, applies the same masking pre-processing,
and saves the same output products.

Usage
-----
    python run_jessnet.py                      \\
        --input  /nas/HI21cm/input_data/       \\
        --output /nas/bonjean/jessnet_results/ \\
        --weights /nas/HI21cm/learnlet_weights/jessnet_learnlet.pth \\
        --nside  256  --n_comp 4  --nscales 5  \\
        --thresh percentile                     \\
        --bright_mask  --strategy_mask 0

Set --thresh to 'percentile' or 'ksigma' to switch between the two
JESSNet variants.
"""

import argparse
import os
import time
import h5py
import numpy as np
import healpy as hp
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
import scipy.fftpack
from tqdm import tqdm
from astropy.io import fits

from jessnet import JESSNet
import healpytools as hpyt


# ---------------------------------------------------------------------------
# Beam helpers (kept from original run script)
# ---------------------------------------------------------------------------

def theta_FWHM(nu, dish_diam):
    """FWHM in radians. nu in MHz, dish_diam in metres."""
    c = 3.0e8
    return c * 1e-6 / nu / float(dish_diam)


def getBeam(theta_FWHM_rad, lmax):
    """Gaussian beam Bl."""
    sigma_b = theta_FWHM_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1, dtype=float)
    return np.exp(-ell * (ell + 1) * sigma_b**2 / 2.0)


def gen_beam_model(obs_maps, freqs, nside, degraded=False, oscillating=False,
                   modairy=False, no_beam=False,
                   A_beam=0.5, T_beam=20,
                   inputbeampath='/home/gkogkou/docs/HI_21cm/HI_21cm_shared/data/'):
    lmax = 3 * nside
    D_telescope = 13.5
    th = theta_FWHM(freqs, D_telescope)

    if no_beam:
        return np.ones_like(th), np.arange(lmax + 1, dtype=float), \
               np.ones((len(freqs), lmax + 1))

    if modairy:
        Beam_l = np.load(inputbeampath + 'beam_marta_modairy.npy')[:len(freqs)]
        f_interp = [
            interp1d(np.arange(len(Beam_l[i])) * 3.4, Beam_l[i],
                     bounds_error=False, fill_value=0)
            for i in range(len(freqs))
        ]
        beam_model = np.array([f(np.arange(lmax + 1)) for f in f_interp])
        return th, np.arange(lmax + 1, dtype=float), beam_model

    if degraded:
        th = np.ones_like(freqs) * th.max()
    if oscillating:
        th += np.deg2rad(A_beam / 60.0) * np.sin(2 * np.pi * freqs / T_beam)

    beam_model = np.array([getBeam(t, lmax) for t in th])
    return th, np.arange(lmax + 1, dtype=float), beam_model


# ---------------------------------------------------------------------------
# Bright-source masking (kept from original run script)
# ---------------------------------------------------------------------------

def masking_polyn(mock_cube, freqs, n=6, sigma_mask=5):
    """Polynomial-fitting bright-source masking."""
    print('  Masking bright HI sources (polynomial fitting) ...')
    rec_polyn = np.copy(mock_cube)
    filtered  = np.copy(mock_cube)
    smooth_fit = np.zeros_like(mock_cube)

    for i in range(mock_cube.shape[1]):
        z = np.polyfit(freqs, mock_cube[:, i], n)
        rec_polyn[:, i] = mock_cube[:, i] - np.polyval(z, freqs)

    stds = np.zeros(rec_polyn.shape[1])
    means = np.zeros_like(stds)
    for i in range(rec_polyn.shape[1]):
        clipped = sigmaclip(rec_polyn[:, i], 5).clipped
        stds[i]  = np.std(clipped)
        means[i] = np.mean(clipped)

    mask = np.abs(rec_polyn - means[None, :]) < sigma_mask * stds[None, :]

    for i in range(mock_cube.shape[1]):
        z = np.polyfit(freqs[mask[:, i]], mock_cube[:, i][mask[:, i]], n)
        smooth_fit[:, i] = np.polyval(z, freqs)

    for ifr in range(freqs.shape[0]):
        idx = np.where(~mask[ifr])[0]
        filtered[ifr, idx] = smooth_fit[ifr, idx]

    return filtered


def _starlet_1d(signal, J=4):
    from scipy.ndimage import convolve1d
    h = np.array([1, 4, 6, 4, 1]) / 16.0
    approx, coeffs = signal.copy(), []
    for j in range(J):
        step = 2 ** j
        h_j = np.zeros((len(h) - 1) * step + 1)
        h_j[::step] = h
        smoothed = convolve1d(approx, h_j, mode='reflect')
        coeffs.append(approx - smoothed)
        approx = smoothed
    coeffs.append(approx)
    return coeffs


def masking_wavelet(mock_cube, sigma_mask=5):
    """Wavelet bright-source masking along frequency axis."""
    print('  Masking bright HI sources (wavelets) ...')
    filtered = np.copy(mock_cube)
    for i in range(mock_cube.shape[1]):
        coeffs = np.array(_starlet_1d(mock_cube[:, i]))
        for sc in range(2):
            trim = 2 ** (sc + 1)
            sub = coeffs[sc][trim:-trim]
            sigma = 1.4826 * np.median(np.abs(sub - np.median(sub)))
            sub[sub > sigma_mask * sigma] = 0
            coeffs[sc][trim:-trim] = sub
        filtered[:, i] = np.sum(coeffs, axis=0)
    return filtered


# ---------------------------------------------------------------------------
# Power spectrum helpers
# ---------------------------------------------------------------------------

def Cell(m):
    cl = hp.anafast(m)
    return np.arange(len(cl)), cl


def _clustering_nu(field, idx_los, nu_ch):
    T = field[:, idx_los]
    dnu = abs(nu_ch[-1] - nu_ch[-2])
    dims = len(nu_ch)
    deltaT = np.array([T[:, i] for i in range(len(idx_los))])
    delta_k = scipy.fftpack.fftn(deltaT, overwrite_x=True, axes=1) * dnu
    return dims, dnu, np.abs(delta_k)**2


def plot_nuPk(fmap, idx_los, nu_ch):
    dims, dnu, dk2 = _clustering_nu(fmap, idx_los, nu_ch)
    middle = dims // 2
    modes = np.arange(dims, dtype=float)
    modes[modes > middle] -= dims
    k = np.abs(modes) * (2.0 * np.pi / (dnu * dims))
    k_bins = np.linspace(0, middle, middle + 1) * (2.0 * np.pi / (dnu * dims))
    k_modes = np.histogram(k, bins=k_bins)[0]
    k_bin   = np.histogram(k, bins=k_bins, weights=k)[0] / k_modes
    Pk      = np.histogram(k, bins=k_bins, weights=np.mean(dk2, axis=0))[0]
    Pk      = Pk / (dnu * dims * k_modes)
    return k_bin[1:], Pk[1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    start = time.time()
    os.makedirs(args.output, exist_ok=True)
    print(f'Results will be saved to: {args.output}')

    # ---- Load data ----
    print('Loading data ...')
    data_file = h5py.File(
        os.path.join(args.input,
                     f'sim_1MHz_nside{args.nside}_gaussian.hd5'), 'r'
    )
    freqs    = np.array(data_file['frequencies'])
    obs_maps = np.array(data_file['Obs_conv_noise'])
    hi_gt    = np.array(data_file['HI_conv_noise'])
    data_file.close()

    obs_maps -= np.mean(obs_maps, axis=1, keepdims=True)
    hi_gt    -= np.mean(hi_gt,    axis=1, keepdims=True)

    print(f'  {len(freqs)} channels from {freqs.min():.1f} to {freqs.max():.1f} MHz')
    print(f'  nside = {args.nside}')

    # ---- Beam model ----
    _, _, beam_model = gen_beam_model(
        obs_maps, freqs, args.nside,
        degraded=False, oscillating=False,
        modairy=(args.beam == 'modairy'),
    )

    # ---- Bright source masking ----
    if args.bright_mask:
        if args.strategy_mask == 0:
            obs_masked = masking_polyn(obs_maps, freqs, sigma_mask=args.sigma_mask)
        else:
            obs_masked = masking_wavelet(obs_maps, sigma_mask=args.sigma_mask)
    else:
        obs_masked = obs_maps.copy()

    # ---- Galactic plane mask ----
    if args.galmask_path:
        planck_mask = fits.getdata(args.galmask_path)
        galmask = hp.ud_grade(
            hp.reorder(planck_mask['GAL070'], n2r=True), args.nside
        ).astype(bool)
    else:
        galmask = np.ones(obs_maps.shape[1], dtype=bool)

    # ---- Run JESSNet ----
    weight_paths = (
        args.weights if isinstance(args.weights, list) else [args.weights]
    )
    if len(weight_paths) == 1:
        weight_paths = weight_paths[0]   # single string → same net for all

    jess = JESSNet(
        obs_masked,
        beam_model,
        n=args.n_comp,
        weight_path=weight_paths,
        thresh=args.thresh,
        nscales=args.nscales,
        nbItMin=args.nbItMin,
        nside_cut=args.nside_cut,
        c_wu=np.array([args.c_wu_max, args.c_wu_min]),
        galmask=galmask,
        verb=1,
    )
    jess.run()

    # Foreground-subtracted residual
    RecHI = obs_maps - jess.fgd_convolved
    RecHI -= np.mean(RecHI, axis=1, keepdims=True)

    A = jess.A
    S = jess.S

    print('Foreground removal complete.')

    # ---- Power spectra ----
    print('Computing power spectra ...')
    n_pix  = hp.nside2npix(args.nside)
    idx_los_full = np.arange(n_pix)

    k_hi,  P_hi  = plot_nuPk(hi_gt,  idx_los_full, freqs)
    k_rec, P_rec = plot_nuPk(RecHI,  idx_los_full, freqs)

    cl_hi  = np.zeros((len(freqs), 3 * args.nside))
    cl_rec = np.zeros_like(cl_hi)
    for nu_i in tqdm(range(len(freqs)), desc='Angular Cl'):
        ell, cl_hi[nu_i]  = Cell(hi_gt[nu_i])
        ell, cl_rec[nu_i] = Cell(RecHI[nu_i])

    # ---- Save ----
    print(f'Saving to {args.output} ...')
    out_hdf = os.path.join(args.output, 'reconstructed_HI.hd5')
    with h5py.File(out_hdf, 'w') as fout:
        fout.create_dataset('RecHI_JESSNet', data=RecHI)
        fout.create_dataset('A_JESSNet',     data=A)
        fout.create_dataset('S_JESSNet',     data=S)
        fout.create_dataset('thresh',        data=args.thresh)
        fout.create_dataset('n_comp',        data=args.n_comp)

    np.save(os.path.join(args.output, 'freq_powspec_JESSNet.npy'),
            np.stack([k_rec, P_rec], axis=0))
    np.save(os.path.join(args.output, 'freq_powspec_HI.npy'),
            np.stack([k_hi, P_hi], axis=0))
    np.save(os.path.join(args.output, 'angular_powspec_JESSNet.npy'), cl_rec)
    np.save(os.path.join(args.output, 'angular_powspec_HI.npy'),      cl_hi)
    np.save(os.path.join(args.output, 'ells.npy'), ell)

    elapsed = (time.time() - start) / 60
    print(f'Done in {elapsed:.1f} min. Results saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run JESSNet HI foreground removal.'
    )
    parser.add_argument('--input',    type=str, required=True,
                        help='Directory with sim_1MHz_nside*.hd5 files')
    parser.add_argument('--output',   type=str, required=True,
                        help='Output directory')
    parser.add_argument('--weights',  type=str, nargs='+', required=True,
                        help='Path(s) to Learnlet .pth weight file(s). '
                             'Pass one path to share across components, '
                             'or n_comp paths for component-specific nets.')
    parser.add_argument('--nside',    type=int, default=256)
    parser.add_argument('--n_comp',   type=int, default=4,
                        help='Number of foreground components (default: 4)')
    parser.add_argument('--nscales',  type=int, default=5)
    parser.add_argument('--nside_cut', type=int, default=16)
    parser.add_argument('--thresh',   type=str, default='percentile',
                        choices=['percentile', 'ksigma'])
    parser.add_argument('--nbItMin',  type=int, default=100)
    parser.add_argument('--c_wu_max', type=float, default=1e-2)
    parser.add_argument('--c_wu_min', type=float, default=1e-3)
    parser.add_argument('--bright_mask', action='store_true',
                        help='Apply bright-source masking')
    parser.add_argument('--strategy_mask', type=int, default=0,
                        choices=[0, 1],
                        help='0=polynomial, 1=wavelet (default: 0)')
    parser.add_argument('--sigma_mask', type=float, default=5.0)
    parser.add_argument('--galmask_path', type=str, default=None,
                        help='Path to Planck GAL mask FITS file (optional)')
    parser.add_argument('--beam',     type=str, default='gaussian',
                        choices=['gaussian', 'modairy'])

    args = parser.parse_args()
    main(args)
