import numpy as np
import healpy as hpy
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial


# Spherical harmonic transform

def compute_alm(map, lmax, iter):
    return hpy.sphtfunc.map2alm(map, lmax=lmax, iter=iter)

def map2alm(maps, lmax=None, iter=3):
    """Computes the alm of a Healpix map.

    Parameters
    ----------
    maps: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps in Healpix representation
    lmax: int
        maximum l of the alm (default: 3*nside)
    iter: 3
        number of iterations

    Returns
    -------
    np.ndarray
        (t,) or (n,t) complex array, alm or stack of n alms
    """

    if len(np.shape(maps)) == 1:
        if lmax is None:
            lmax = 3*hpy.get_nside(maps)
        return hpy.sphtfunc.map2alm(maps, lmax=lmax, iter=iter)

    n = np.shape(maps)[0]
    if lmax is None:
        lmax = 3*hpy.get_nside(maps[0, :])
        
    args = [(maps[i], lmax, iter) for i in range(n)]
    with mp.Pool(processes=20) as pool:
        result = pool.starmap(compute_alm, args)

    result = np.array(result)
    #result = np.array([hpy.sphtfunc.map2alm(maps[i, :], lmax=lmax, iter=iter) for i in range(n)])
    return result

def compute_map(alm, nside):
    return hpy.sphtfunc.alm2map(alm, nside)

def alm2map(alms, nside):
    """Computes a Healpix map given the alm.

    Parameters
    ----------
    alms: np.ndarray
        (t,) or (n,t) complex array, alm or stack of n alms
    nside: int
        nside of the output Healpix maps

    Returns
    -------
    np.ndarray
        (p,) or (n,p) float array, map or stack of n maps in Healpix representation
    """

    if len(np.shape(alms)) == 1:
        return hpy.alm2map(alms, nside)#, verbose=False)

    n = np.shape(alms)[0]
    
    #args = [(alms[i], nside) for i in range(n)]
    #with mp.Pool(processes=20) as pool:
    #    result = pool.starmap(compute_map, args)

    #result = np.array(result)
    result = np.array([hpy.sphtfunc.alm2map(alms[i, :], nside) for i in range(n)])
    return result

def smoothalm(alm, filters):
    return hpy.sphtfunc.smoothalm(alm, beam_window=filters, inplace=False)

def alm_product(alms, filters):
    """Apply an isotropic filter on an alm.

    Parameters
    ----------
    alms: np.ndarray
        (t,) or (n,t) complex array, alm or stack of n alms
    filters: np.ndarray
        (lmax+1,) or (n,lmax+1) float array, isotropic filter or stack of n isotropic filters (one filter per source) in
         spherical harmonic domain

    Returns
    -------
    np.ndarray
        (t,) or (n,t) complex array, filtered alm or stack of n filtered alms
    """

    dim_filters = len(np.shape(filters))
    dim_alms = len(np.shape(alms))

    if dim_filters == 1 and dim_alms == 1:
        return hpy.sphtfunc.smoothalm(alms, beam_window=filters, verbose=False, inplace=False)

    n = np.shape(alms)[0]

    if dim_filters == 1:
        #args = [(alms[i], filters) for i in range(n)]
        #with mp.Pool(processes=20) as pool:
        #    result = pool.starmap(smoothalm, args)
        #    result = np.array(result)
        result = np.array([hpy.sphtfunc.smoothalm(alms[i, :], beam_window=filters, inplace=False) for i in range(n)])
        return result
    else:
        #args = [(alms[i], filters[i]) for i in range(n)]
        #with mp.Pool(processes=20) as pool:
        #    result = pool.starmap(smoothalm, args)
        #    result = np.array(result)
        result = np.array([hpy.sphtfunc.smoothalm(alms[i, :], beam_window=filters[i, :], inplace=False) for i in range(n)])
        return result


def convolve(maps, filters, lmax=None, nside=None):
    """Convolve maps with filters.

    Parameters
    ----------
    maps: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps in Healpix representation
    filters: np.ndarray
        (lmax+1,) or (n,lmax+1) float array, isotropic filter or stack of n isotropic filters (one filter per source)
    lmax: int
        maximum l of the filtering (default: deduced from filters)
    nside: int
        nside of the output Healpix maps (default: deduced from maps)

    Returns
    -------
    maps: np.ndarray
        (p,) or (n,p) float array, convolved map or stack of n convolved maps in Healpix representation
    """

    if lmax is not None:
        if len(np.shape(filters)) == 1:
            lmax = len(filters) - 1
        else:
            lmax = np.shape(filters)[1] - 1

    alms = map2alm(maps, lmax=lmax)

    alms = alm_product(alms, filters)

    if nside is None:
        nside = hpy.get_nside(maps)

    return alm2map(alms, nside=nside)


def anafast(maps, lmax=None, iter=3):
    """Computes the angular power spectrum of a Healpix map.

    Parameters
    ----------
    maps: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps in Healpix representation
    lmax: int
        maximum l of the angular power spectrum (default: 3*nside of maps)
    iter: 3
        number of iterations

    Returns
    -------
    np.ndarray
        (lmax+1,) or (n,lmax+1) float array, angular power spectrum or stack of n angular power spectra
    """

    if len(np.shape(maps)) == 1:
        if lmax is None:
            lmax = 3*hpy.get_nside(maps)
        return hpy.sphtfunc.anafast(maps, lmax=lmax, iter=iter)

    n = np.shape(maps)[0]
    if lmax is None:
        lmax = 3 * hpy.get_nside(maps[0, :])
    return np.array([hpy.sphtfunc.anafast(maps[i, :], lmax=lmax) for i in range(n)])


def alm2cl(alms):
    """Computes the angular power spectrum from an alm.

    Parameters
    ----------
    alms: np.ndarray
        (t,) or (n,t) complex array, alm or stack of n alms

    Returns
    -------
    np.ndarray
        (lmax+1,) or (n,lmax+1) float array, angular power spectrum or stack of n angular power spectra
    """

    if len(np.shape(alms)) == 1:
        return hpy.sphtfunc.alm2cl(alms)

    n = np.shape(alms)[0]
    return np.array([hpy.sphtfunc.alm2cl(alms[i, :]) for i in range(n)])


# Alm index computation

def getsize(lmax):
    """Returns the size of the array needed to store alm up to lmax.

    Parameters
    ----------
    lmax: int
        maximum l of the alm

    Returns
    -------
    int
        size of the array needed to store alm up to lmax

    """

    return hpy.Alm.getsize(lmax)


def getlm(lmax):
    """Get the mapping of an alm.

    Parameters
    ----------
    lmax: int
        maximum l of the alm

    Returns
    -------
    (np.ndarray,np.ndarray)
        l to index map,
        m to index map
    """

    return hpy.Alm.getlm(lmax)


def npix2nside(npix):
    """
    Give the nside parameter for the given number of pixels.

    Parameters
    ----------
    npix: int
        number of pixels

    Returns
    -------
    nside: int
        nside
    """

    return hpy.npix2nside(npix)


# Wavelet filtering

def spline2(size, l, lc):
    """
    Compute a non-negative decreasing spline, with value 1 at index 0.

    Parameters
    ----------
    size: int
        size of the spline
    l: float
        spline parameter
    lc: float
        spline parameter

    Returns
    -------
    np.ndarray
        (size,) float array, spline
    """

    res = np.arange(0, size+1)
    res = 2*l*res/(lc*size)
    res = (3/2) * 1/12 * (abs(res-2)**3 - 4*abs(res-1)**3 + 6*abs(res)**3 - 4*abs(res+1)**3 + abs(res+2)**3)
    return res


def compute_h(size, lc):
    """
    Compute a low-pass filter.

    Parameters
    ----------
    size: int
        size of the filter
    lc: float
        cutoff parameter

    Returns
    -------
    np.ndarray
        (size,) float array, filter
    """

    tab1 = spline2(size, 2*lc, 1)
    tab2 = spline2(size, lc, 1)
    h = tab1/(tab2+1e-6)
    h[int(size/(2*lc)):size] = 0
    return h


def compute_g(size, lc):
    """
    Compute a high-pass filter.

    Parameters
    ----------
    size: int
        size of the filter
    lc: float
        cutoff parameter

    Returns
    -------
    np.ndarray
        (size,) float array, filter
    """

    tab1 = spline2(size, 2*lc, 1)
    tab2 = spline2(size, lc, 1)
    g = (tab2-tab1)/(tab2+1e-6)
    g[int(size/(2*lc)):size] = 1
    return g


def get_wt_filters(lmax, nscales):
    """Compute wavelet filters.

    Parameters
    ----------
    lmax: int
        maximum l
    nscales: int
        number of wavelet detail scales

    Returns
    -------
    np.ndarray
        (lmax+1,nscales+1) float array, filters
    """

    wt_filters = np.ones((lmax+1, nscales+1))
    wt_filters[:, 1:] = np.array([compute_h(lmax, 2**scale) for scale in range(nscales)]).T
    wt_filters[:, :nscales] -= wt_filters[:, 1:(nscales+1)]
    return wt_filters


def wt_trans(inputs, nscales=3, lmax=None, alm_in=False, nside=None, alm_out=False):
    """Wavelet transform an array.

    Parameters
    ----------
    inputs: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps / if alm_in, (t,) or (n,t) complex array, alm or stack
        of n alms
    nscales: int
        number of wavelet detail scales
    lmax: int
        maximum l (default: 3*nside / if alm_in, deduced from inputs)
    alm_in: bool
        inputs is alm
    nside: int
        nside of the output Healpix maps (default: deduced from maps)
    alm_out: bool
        output is alm

    Returns
    -------
    np.ndarray
        (p,nscales+1) or (n,p,scales+1) float array, wavelet transform of the input array or stack of the wavelet
        transforms of the n input arrays / if alm_out, (t,nscales+1) or (n,t,scales+1) complex array, alm of the
        wavelet transform of the input array or stack of the alms of the wavelet transforms of the n input arrays
    """

    dim_inputs = len(np.shape(inputs))
    maps = None  # to remove warnings

    if alm_in:
        alms = inputs
        if nside is None and not alm_out:
            raise ValueError("nside is missing")
        if not alm_out:
            maps = alm2map(alms, nside)
        if lmax is None:
            lmax = hpy.Alm.getlmax(np.shape(alms)[-1])

    else:
        maps = inputs
        if dim_inputs == 1:
            nside = hpy.get_nside(maps)
        else:
            nside = hpy.get_nside(maps[0, :])
        if lmax is None:
            lmax = 3 * nside
        alms = map2alm(maps, lmax=lmax)

    if not alm_out:
        l_scale = maps.copy()
        if dim_inputs == 1:
            npix = len(maps)
            wts = np.zeros((npix, nscales + 1))
        else:
            npix = np.shape(maps)[1]
            wts = np.zeros((np.shape(maps)[0], npix, nscales + 1))
    else:
        l_scale = alms.copy()
        if dim_inputs == 1:
            npix = np.size(alms)
            wts = np.zeros((npix, nscales + 1), dtype='complex')
        else:
            npix = np.shape(alms)[1]
            wts = np.zeros((np.shape(maps)[0], npix, nscales + 1), dtype='complex')

    scale = 1
    for j in range(nscales):
        h = compute_h(lmax, scale)
        if not alm_out:
            m = alm2map(alm_product(alms, h), nside)
        else:
            m = alm_product(alms, h)
        h_scale = l_scale - m
        l_scale = m
        if dim_inputs == 1:
            wts[:, j] = h_scale
        else:
            wts[:, :, j] = h_scale
        scale *= 2

    if dim_inputs == 1:
        wts[:, nscales] = l_scale
    else:
        wts[:, :, nscales] = l_scale

    return wts


def wt_rec(wts):
    """Reconstruct a wavelet decomposition.

    Parameters
    ----------
    wts: np.ndarray
        (p,nscales+1) or (n,p,scales+1) float array, wavelet transform of a map or stack of the wavelet transforms of n
        maps

    Returns
    -------
    np.ndarray
        (p,) or (n,p,) float array, reconstructed map or stack of n reconstructed maps
    """

    return np.sum(wts, axis=-1)


# Plots

def mollview(maps, maps2=None, log=False, unit='', title='', minimum=None, maximum=None, cbar=True):
    """Plot one or more Healpix maps in Mollweide projection.

    Parameters
    ----------
    maps: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps
    maps2: np.ndarray
        (p,) or (n,p) float array, second map or stack of n maps, optional
    log: bool
        logarithmic scale
    unit: str
        unit of the data
    title: str
        title of the plots
    minimum: float
        minimum range value (default: min(maps, maps2))
    maximum: float
        maximum range value (default: max(maps, maps2))
    cbar: bool
        show color bar

    Returns
    -------
    None
    """

    if len(np.shape(maps)) == 1:
        maps = np.expand_dims(maps, axis=0)
        if maps2 is not None:
            maps2 = np.expand_dims(maps2, axis=0)
    if minimum is None:
        minimum = np.min(maps)
        if maps2 is not None:
            minimum = np.min([minimum, np.min(maps2)])
    if maximum is None:
        maximum = np.max(maps)
        if maps2 is not None:
            maximum = np.max([maximum, np.max(maps2)])
    if not log:
        def f(x): return x
    else:
        def f(x): return np.log10(x - minimum + 1)
    for i in range(np.shape(maps)[0]):
        hpy.mollview(f(maps[i, :]), fig=None, unit=unit, title=title, min=f(minimum), max=f(maximum), cbar=cbar)
        if maps2 is not None:
            hpy.mollview(f(maps2[i, :]), fig=None, unit=unit, title=title, min=f(minimum), max=f(maximum), cbar=cbar)


def view_spec(inputs, lmax=None, alm_in=False):
    """Plot the angular power spectrum of one or several maps.

    Parameters
    ----------
    inputs: np.ndarray
        (p,) or (n,p) float array, map or stack of n maps / if alm_in, (t,) or (n,t) complex array, alm or stack
        of n alms
    lmax: int
        maximum l (default: 3*nside / if alm_in, deduced from inputs)
    alm_in: bool
        inputs is alm

    Returns
    -------
    None
    """

    if len(np.shape(inputs)) == 1:
        inputs = np.expand_dims(inputs, axis=0)

    if not alm_in:
        cls = anafast(inputs, lmax=lmax)
    else:
        cls = alm2cl(inputs)

    plt.figure()
    for i in range(np.shape(inputs)[0]):
        plt.semilogy(cls[i, :], label='Source '+str(i+1))
    plt.xlabel('$l$')
    plt.ylabel('$c_l$')
    if np.shape(inputs)[0] != 1:
        plt.legend()


# Miscellaneous

def getidealbeam(lmax, cutmin=None, cutmax=None):
    """Compute a beam, with value 1 until a first cutoff frequency and 0 after a second cutoff frequency. The transition
    is computed with a spline.

    Parameters
    ----------
    lmax: int
        maximum l
    cutmin: int
        frequency below which filter is 1 (default: int((lmax+1)/4))
    cutmax: int
        frequency above which filter is 0 (default: int((lmax+1)/2))

    Returns
    -------
    np.ndarray
        (lmax+1,) float array, filter
    """

    if cutmin is None:
        cutmin = int((lmax+1)/4)
    if cutmax is None:
        cutmax = int((lmax+1)/2)
    bl = np.zeros(lmax+1)
    bl[0:cutmin] = 1
    bl[cutmin:cutmax] = spline2(cutmax-cutmin-1, 1, 1)
    return bl


def getbeam(fwhm=100, lmax=512):
    """Get a spherical Gaussian-shaped beam.

    Parameters
    ----------
    fwhm: float
        full width at half maximum in the harmonic space (in terms of l)
    lmax: int
        maximum l

    Returns
    -------
    np.ndarray
        (lmax+1,) float array, Gaussian-shaped beam
    """
    
    tor = 0.0174533
    if len(np.shape(fwhm)) == 1:
        fwhm = np.expand_dims(fwhm, axis=1)
    F = fwhm / 60 * tor
    l = np.arange(0, lmax+1)
    ell = l * (l + 1)
    bl = np.exp(-ell * F * F / 16 / np.log(2))
    return bl

"""
wt_trans accéléré avec ducc0 0.40
"""

import numpy as np
import ducc0.sht as _sht
import ducc0.healpix as _hpx
import healpy as hpy

# ──────────────────────────────────────────────────────────────
# Géométrie HEALPix (mise en cache par nside)
# ──────────────────────────────────────────────────────────────
_geom_cache = {}

def _get_geom(nside):
    if nside in _geom_cache:
        return _geom_cache[nside]
    base  = _hpx.Healpix_Base(nside, "RING")
    info  = base.sht_info()
    theta     = np.asarray(info["theta"],     dtype=np.float64)
    phi0      = np.asarray(info["phi0"],      dtype=np.float64)
    nphi      = np.asarray(info["nphi"],      dtype=np.uint64)
    ringstart = np.asarray(info["ringstart"], dtype=np.uint64)

    nrings = len(theta)
    unit   = 4 * np.pi / (3.0 * nside**2)
    pixel_w = np.empty(12 * nside**2, dtype=np.float64)
    for r in range(nrings):
        i  = r + 1
        if   i < nside:          w = unit * i
        elif i <= 3 * nside:     w = unit * nside
        else:                    w = unit * (4 * nside - i)
        s, n = int(ringstart[r]), int(nphi[r])
        pixel_w[s:s+n] = w / n

    _geom_cache[nside] = (theta, phi0, nphi, ringstart, pixel_w)
    return _geom_cache[nside]


# ──────────────────────────────────────────────────────────────
# l_idx cache : ordre healpy des alm
# healpy stocke par m croissant :
#   m=0 : l=0,1,2,...,lmax
#   m=1 :   l=1,2,...,lmax
#   ...
# ──────────────────────────────────────────────────────────────
_lidx_cache = {}

def _get_l_idx(lmax):
    if lmax in _lidx_cache:
        return _lidx_cache[lmax]
    l, _ = hpy.Alm.getlm(lmax)
    _lidx_cache[lmax] = l
    return l


# ──────────────────────────────────────────────────────────────
# map2alm  (batch)
# ──────────────────────────────────────────────────────────────
def map2alm_ducc(maps, lmax, nthreads=8):
    """
    maps : (npix,) ou (n, npix) float64
    retourne : (nalm,) ou (n, nalm) complex128
    """
    nside = hpy.get_nside(maps) if maps.ndim == 1 else hpy.get_nside(maps[0])
    theta, phi0, nphi, ringstart, pixel_w = _get_geom(nside)

    def _one(m):
        mw = np.ascontiguousarray((m * pixel_w)[np.newaxis, :], dtype=np.float64)
        return _sht.adjoint_synthesis(
            map=mw, theta=theta, nphi=nphi, phi0=phi0,
            ringstart=ringstart, lmax=lmax, mmax=lmax,
            spin=0, nthreads=nthreads,
        )[0]

    if maps.ndim == 1:
        return _one(maps)
    return np.array([_one(maps[i]) for i in range(maps.shape[0])])


# ──────────────────────────────────────────────────────────────
# alm2map  (batch)
# ──────────────────────────────────────────────────────────────
def alm2map_ducc(alms, nside, nthreads=8):
    """
    alms : (nalm,) ou (n, nalm) complex128
    retourne : (npix,) ou (n, npix) float64
    """
    theta, phi0, nphi, ringstart, _ = _get_geom(nside)
    lmax = hpy.Alm.getlmax(alms.shape[-1])

    def _one(a):
        a2d = np.ascontiguousarray(a[np.newaxis, :])
        return _sht.synthesis(
            alm=a2d, theta=theta, nphi=nphi, phi0=phi0,
            ringstart=ringstart, lmax=lmax, mmax=lmax,
            spin=0, nthreads=nthreads,
        )[0]

    if alms.ndim == 1:
        return _one(alms)
    return np.array([_one(alms[i]) for i in range(alms.shape[0])])


# ──────────────────────────────────────────────────────────────
# alm_product  — utilise l_idx dans l'ordre healpy exact
# ──────────────────────────────────────────────────────────────
def alm_product_ducc(alms, filters):
    """
    alms    : (nalm,) ou (n, nalm) complex128
    filters : (lmax+1,) ou (n, lmax+1) float64
    """
    lmax  = hpy.Alm.getlmax(alms.shape[-1])
    l_idx = _get_l_idx(lmax)    # ordre healpy correct : m varie lentement

    if alms.ndim == 1:
        return alms * filters[l_idx]

    if filters.ndim == 1:
        return alms * filters[l_idx][np.newaxis, :]
    else:
        return alms * filters[:, l_idx]


# ──────────────────────────────────────────────────────────────
# wt_trans  (version ducc0)
# ──────────────────────────────────────────────────────────────
def wt_trans_ducc(inputs, nscales=3, lmax=None, alm_in=False,
                  nside=None, alm_out=False, nthreads=8):
    """
    Transformée en ondelettes sphériques (à trous), accélérée ducc0.
    Interface identique à hpyt.wt_trans + paramètre nthreads.
    """
    dim_inputs = inputs.ndim
    maps = None

    if alm_in:
        alms = inputs
        if nside is None and not alm_out:
            raise ValueError("nside requis quand alm_in=True et alm_out=False")
        if not alm_out:
            maps = alm2map_ducc(alms, nside, nthreads=nthreads)
        if lmax is None:
            lmax = hpy.Alm.getlmax(alms.shape[-1])
    else:
        maps = inputs
        nside = hpy.get_nside(maps) if dim_inputs == 1 else hpy.get_nside(maps[0])
        if lmax is None:
            lmax = 3 * nside
        alms = map2alm_ducc(maps, lmax, nthreads=nthreads)

    if not alm_out:
        l_scale = maps.copy()
        if dim_inputs == 1:
            wts = np.zeros((maps.shape[0], nscales + 1), dtype=np.float64)
        else:
            wts = np.zeros((maps.shape[0], maps.shape[1], nscales + 1), dtype=np.float64)
    else:
        l_scale = alms.copy()
        nalm = alms.shape[-1]
        if dim_inputs == 1:
            wts = np.zeros((nalm, nscales + 1), dtype=np.complex128)
        else:
            wts = np.zeros((alms.shape[0], nalm, nscales + 1), dtype=np.complex128)

    scale = 1
    for j in range(nscales):
        h = compute_h(lmax, scale)
        filtered_alms = alm_product_ducc(alms, h)

        if not alm_out:
            m = alm2map_ducc(filtered_alms, nside, nthreads=nthreads)
        else:
            m = filtered_alms

        h_scale = l_scale - m
        l_scale = m

        if dim_inputs == 1:
            wts[:, j] = h_scale
        else:
            wts[:, :, j] = h_scale

        scale *= 2

    if dim_inputs == 1:
        wts[:, nscales] = l_scale
    else:
        wts[:, :, nscales] = l_scale

    return wts


# ──────────────────────────────────────────────────────────────
# Vérification de l_idx
# ──────────────────────────────────────────────────────────────
def check_l_idx(lmax=10):
    l_idx = _get_l_idx(lmax)
    l_hp, m_hp = hpy.Alm.getlm(lmax)
    nalm = hpy.Alm.getsize(lmax)
    for i in range(nalm):
        assert l_idx[i] == l_hp[i], f"mismatch i={i}: l_idx={l_idx[i]}, healpy={l_hp[i]}"
    print(f"l_idx OK ({nalm} coefficients vérifiés pour lmax={lmax})")
