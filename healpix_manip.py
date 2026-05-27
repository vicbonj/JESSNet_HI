import numpy as np
import healpy as hp
from astropy import wcs
from tqdm import tqdm
from skimage.transform import resize


def make_wcs(size, glon, glat, reso, rot_rand):
    w = wcs.WCS(naxis = 2)
    w.wcs.crpix = [(size-1)/2., (size-1)/2.]
    w.wcs.crval = [glon, glat]
    w.wcs.cdelt = [-reso, reso]
    w.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    w.wcs.crota = [-rot_rand, -rot_rand]
    return w

def map_proj(w):
    size = int(w.wcs.crpix[0]*2+1)
    yy, xx = np.indices((size, size))
    patch_GLON, patch_GLAT = w.wcs_pix2world(xx, yy, 1)
    return patch_GLON, patch_GLAT

def extract_proj(maps, patch_GLON, patch_GLAT, nside, interp):
    p_PHI = patch_GLON*np.pi/180.
    p_TH = patch_GLAT*np.pi/180.
    if interp is True:
        patch = [hp.get_interp_val(m, np.pi/2.-p_TH, p_PHI) for m in maps]
    else:
        pix = hp.ang2pix(nside, np.pi/2-p_TH, p_PHI)
        patch = [m[pix] for m in maps]
    return patch

def from_healpix_to_maps(maps, nside=2048, nside_cut=16, npix=256+16*2, reso=1.5/60, interp=False):
    patches = []
    for i in tqdm(range(hp.nside2npix(nside_cut))):
        lon, lat = hp.pix2ang(nside_cut, i, lonlat=True)
        if abs(lat) <= 45:
            rot = 45
        elif (abs(lat) > 45) & (abs(lat) <= 60):
            rot = 45
        elif (abs(lat) > 60) & (abs(lat) <= 84):
            rot = 45# * (i%2)
        else:
            rot = 45
        #if cut is not False:
        #    pixels = npix+cut*2
        #else:
        pixels = npix
        w = make_wcs(pixels, lon, lat, reso, rot)
        p_GLON, p_GLAT = map_proj(w)
        patch = extract_proj(maps, p_GLON, p_GLAT, nside, interp=interp)
        patches.append(patch)
    return patches

def from_maps_to_healpix(patches, nside=2048, nside_cut=16, npix=256+16*2, reso=1.5/60, cut=24, fact=2):
    rec_map = np.zeros((len(patches[0]), hp.nside2npix(nside)))
    counts = np.zeros(hp.nside2npix(nside))
    
    remove = int(fact/2)
    if cut is not False:
        pixels = npix
        rem = remove + cut*fact
    else:
        pixels = npix
        rem = remove
    for i in tqdm(range(hp.nside2npix(nside_cut))):
        lon, lat = hp.pix2ang(nside_cut, i, lonlat=True)
        if abs(lat) <= 45:
            rot = 45
        elif (abs(lat) > 45) & (abs(lat) <= 60):
            rot = 45
        elif (abs(lat) > 60) & (abs(lat) <= 84):
            rot = 45# * (i%2)
        else:
            rot = 45
        w = make_wcs(pixels, lon, lat, reso, rot)
        patch_GLON, patch_GLAT = map_proj(w)
        
        if (np.min(patch_GLON) < 1) & (np.max(patch_GLON) > 359):
            patch_GLON[patch_GLON < 180] += 360
            
        if fact > 1:
            patch_GLON_ok = resize(patch_GLON, (npix*fact, npix*fact))[rem:-rem,rem:-rem]%360
            patch_GLAT_ok = resize(patch_GLAT, (npix*fact, npix*fact))[rem:-rem,rem:-rem]
            patch_ok = [resize(p, (len(p[0])*fact, len(p[0])*fact), order=0)[remove:-remove,remove:-remove] for p in patches[i]]
        else:
            if rem > 0:
                patch_GLON_ok = patch_GLON[rem:-rem,rem:-rem]%360
                patch_GLAT_ok = patch_GLAT[rem:-rem,rem:-rem]
            else:
                patch_GLON_ok = patch_GLON%360
                patch_GLAT_ok = patch_GLAT
                patch_ok = patches[i]
            
        pix = hp.ang2pix(nside, patch_GLON_ok.ravel(), patch_GLAT_ok.ravel(), lonlat=True)

        for j in range(len(rec_map)):
            its = patch_ok[j].ravel()
            for k in range(len(pix)):
                rec_map[j][pix[k]] += its[k]
                if j == 0:
                    counts[pix[k]] += 1

    for j in range(len(rec_map)):
        rec_map[j] = rec_map[j]/counts

    return rec_map, counts

def populate_diagonal_by_diagonal2(to_populate):
    n = int(np.sqrt(len(to_populate)))
    new_array = np.zeros((n, n))
    k = 0
    
    # Diagonales partant du haut
    for col in range(n):
        length = col + 1
        idx = np.arange(length)
        new_array[idx, col - idx] = to_populate[k:k+length]
        k += length

    # Diagonales partant du côté droit (hors première)
    for row in range(1, n):
        length = n - row
        idx = np.arange(length)
        new_array[row + idx, (n-1) - idx] = to_populate[k:k+length]
        k += length

    return new_array

def populate_diagonal_by_diagonal(to_populate):
    n = int(np.sqrt(len(to_populate)))
    new_array = np.zeros((n, n))
    first = [to_populate[np.sum(np.arange(col+1)):np.sum(np.arange(col+1))+col+1] for col in range(n)]
    second = [to_populate[::-1][np.sum(np.arange(col+1)):np.sum(np.arange(col+1))+col+1] for col in range(n-1)]
    for col in range(n):
        i, j = 0, col
        while i < n and j >= 0:
            new_array[i][j] = first[col][i]
            if col < n-1:
                new_array[::-1,::-1][i][j] = second[col][i]
            i += 1
            j -= 1
    return new_array

def from_healpix_to_maps_old(healpix_map, nside_cut=4):
    nb_maps = len(healpix_map)
    nside_map = hp.npix2nside(len(healpix_map[0]))
    conv = hp.ang2pix(nside_cut, *hp.pix2ang(nside_map, np.arange(hp.nside2npix(nside_map))))
    big_pixels = np.zeros(hp.nside2npix(nside_map))
    for i in range(hp.nside2npix(nside_cut)):
        big_pixels[conv == i] = i

    nb_pix = int(np.sqrt(hp.nside2npix(nside_map)/hp.nside2npix(nside_cut)))
    patches = np.zeros((nb_maps, hp.nside2npix(nside_cut), nb_pix, nb_pix))

    #print("Cutting HEALPIX map with Nside={} onto {} patches of {}x{} pixels...".format(nside_map, len(patches[0]), nb_pix, nb_pix))
    for i in range(hp.nside2npix(nside_cut)):
        patch1d = healpix_map[:,big_pixels == i]
        for k in range(nb_maps):
            patch2d = populate_diagonal_by_diagonal2(patch1d[k])
            patches[k,i] = patch2d

    return patches

def parallel_in(args):
    j, i = args
    #i, npix, nside_map, healpix_map = args
    pix = pixs_shared[i]
    patch = healpix_maps_shared[j][pix]
    #print("Worker running")
    return populate_diagonal_by_diagonal2(patch)

def from_healpix_to_maps_new(healpix_maps, nside_cut=4):
    nb_maps = len(healpix_maps)
    nside_map = hp.npix2nside(len(healpix_maps[0]))
    ncoarse_pix = hp.nside2npix(nside_cut)
    npix = len(healpix_maps[0]) / ncoarse_pix
    npix_patch = int(np.sqrt(npix))
    #print("Cutting HEALPIX map with Nside={} onto {} patches of {}x{} pixels...".format(nside_map, ncoarse_pix, npix_patch, npix_patch))
    patches = np.zeros((nb_maps, ncoarse_pix, npix_patch, npix_patch))
    for i in range(ncoarse_pix):
        pix = hp.nest2ring(nside_map, np.arange(i*npix, (i+1)*npix, dtype=int))
        pix.sort()
        for j in range(nb_maps):
            patches[j,i] = populate_diagonal_by_diagonal2(healpix_maps[j,pix])
    return patches

def from_2d_to_diagonal(patch):
    n = len(patch)
    first = [np.diag(patch[:i+1,:i+1][::,::-1]) for i in range(n)]
    second = [np.diag(patch[::-1,::-1][:i+1,:i+1][::,::-1])[::-1] for i in range(n-1)][::-1]
    
    total = []
    for f in first:
        total.extend(f)
    for s in second:
        total.extend(s)

    return np.array(total)

def from_2d_to_diagonal2(patch):
    n = patch.shape[0]
    result = []

    # Diagonales supérieures et principale
    for col in range(n):
        idx = np.arange(col + 1)
        result.append(patch[idx, col - idx])

    # Diagonales inférieures
    for row in range(1, n):
        idx = np.arange(n - row)
        result.append(patch[row + idx, (n-1) - idx])

    return np.concatenate(result)

def from_maps_to_healpix_old(patches):
    nside_out = hp.npix2nside(patches.shape[1]*patches.shape[2]*patches.shape[3])    
    nb_maps = len(patches)
    healpix = np.zeros((nb_maps, hp.nside2npix(nside_out)))
    nside_cut = hp.npix2nside(len(patches[0]))
    conv = hp.ang2pix(nside_cut, *hp.pix2ang(nside_out, np.arange(hp.nside2npix(nside_out))))
    big_pixels = np.zeros(hp.nside2npix(nside_out))
    for i in range(hp.nside2npix(nside_cut)):
        big_pixels[conv == i] = i

    print("Reconstructing onto an HEALPIX map of Nside={}".format(nside_out))
    for i in range(len(patches[0])):
        for k in range(nb_maps):
            patch1d = from_2d_to_diagonal2(patches[k,i])
            healpix[k,big_pixels == i] = patch1d

    return healpix

#import numpy as np
from numba import njit, prange
#import healpy as hp
import numba

numba.set_num_threads(30)  # ou le nombre de cœurs physiques

@njit
def populate_diagonal(to_populate, npix_patch):
    """Remplit un patch 2D à partir d'un tableau 1D en diagonales."""
    n = npix_patch
    patch = np.zeros((n, n), dtype=to_populate.dtype)
    k = 0

    # diagonales supérieures et principale
    for col in range(n):
        for t in range(col+1):
            patch[t, col - t] = to_populate[k]
            k += 1

    # diagonales inférieures
    for row in range(1, n):
        for t in range(n - row):
            patch[row + t, n - 1 - t] = to_populate[k]
            k += 1

    return patch

@njit(parallel=True)
def from_healpix_to_patches_numba(healpix_maps, pixs, ncoarse_pix, npix_patch):
    """Convertit des HEALPix maps en patches 2D parallélisés."""
    nb_maps = healpix_maps.shape[0]
    patches = np.zeros((nb_maps, ncoarse_pix, npix_patch, npix_patch), dtype=healpix_maps.dtype)

    for j in prange(nb_maps):          # boucle parallèle sur les maps
        for i in range(ncoarse_pix):
            flat_patch = healpix_maps[j, pixs[i]]
            patches[j, i] = populate_diagonal(flat_patch, npix_patch)

    return patches

def from_healpix_to_maps_new_numba(healpix_maps, nside_cut=4, verbose=False):
    nb_maps = len(healpix_maps)
    nside_map = hp.npix2nside(len(healpix_maps[0]))
    ncoarse_pix = hp.nside2npix(nside_cut)
    npix = len(healpix_maps[0]) / ncoarse_pix
    npix_patch = int(np.sqrt(npix))

    if verbose:
        print(f"Cutting HEALPix map with Nside={nside_map} "
              f"onto {ncoarse_pix} patches of {npix_patch}x{npix_patch} pixels...")
        
    # Précompute pixs
    pixs = np.array([np.sort(hp.nest2ring(nside_map,
                                           np.arange(i*npix, (i+1)*npix, dtype=np.int64)))
                     for i in range(ncoarse_pix)], dtype=np.int64)

    patches = from_healpix_to_patches_numba(healpix_maps, pixs, ncoarse_pix, npix_patch)
    return patches


@njit
def patch_to_flat(patch):
    """Convertit un patch 2D en tableau 1D selon les diagonales."""
    n = patch.shape[0]
    total_len = n*(n+1)//2*2 - n  # longueur totale diagonales
    result = np.empty(total_len, dtype=patch.dtype)
    k = 0

    # diagonales supérieures et principale
    for col in range(n):
        for t in range(col+1):
            result[k] = patch[t, col - t]
            k += 1

    # diagonales inférieures
    for row in range(1, n):
        for t in range(n - row):
            result[k] = patch[row + t, n - 1 - t]
            k += 1

    return result

@njit(parallel=True)
def from_patches_to_healpix_numba(patches, pixs_out, nb_maps, nb_pix_total):
    ncoarse_pix = patches.shape[1]
    npix_patch = patches.shape[2]
    
    healpix_map = np.zeros((nb_maps, nb_pix_total), dtype=patches.dtype)

    for i in prange(ncoarse_pix):
        pix = pixs_out[i]
        for k in range(nb_maps):
            flat_patch = patch_to_flat(patches[k, i])
            healpix_map[k, pix] = flat_patch

    return healpix_map

def from_maps_to_healpix_new_numba(patches):
    ncoarse_pix = patches.shape[1]
    npix_patch = patches.shape[2]
    nb_maps = patches.shape[0]

    npix_total = ncoarse_pix * npix_patch * npix_patch
    nside_out = hp.npix2nside(npix_total)

    # Précompute pixs_out AVANT Numba
    pixs_out = np.array([np.sort(hp.nest2ring(nside_out,
                                               np.arange(i*npix_patch*npix_patch, 
                                                         (i+1)*npix_patch*npix_patch, dtype=np.int64)))
                         for i in range(ncoarse_pix)], dtype=np.int64)

    # Calculer nb_pix_total pour allouer le tableau
    nb_pix_total = hp.nside2npix(nside_out)

    # Appel Numba avec tous les paramètres connus
    healpix_maps = from_patches_to_healpix_numba(patches, pixs_out, nb_maps, nb_pix_total)
    return healpix_maps

def from_maps_to_healpix_new(patches):
    nside_out = hp.npix2nside(patches.shape[1]*patches.shape[2]*patches.shape[3])
    nb_maps = len(patches)
    npix = patches.shape[2]*patches.shape[3]
    healpix = np.zeros((nb_maps, hp.nside2npix(nside_out)))
    npix_cut = len(patches[0])
    nside_cut = hp.npix2nside(npix_cut)

    #print("Reconstructing onto an HEALPIX map of Nside={}".format(nside_out))
    for i in range(npix_cut):
        pix = hp.nest2ring(nside_out, np.arange(i*npix, (i+1)*npix, dtype=int))
        pix.sort()
        for k in range(nb_maps):
            patch1d = from_2d_to_diagonal2(patches[k,i])
            healpix[k,pix] = patch1d

    return healpix

import numpy as np
import healpy as hp

def find_patch_neighbors(patches, patch_indices, nside_map):

    npatch, h, w = patch_indices.shape
    neighbors = {i: {"left": None, "right": None, "top": None, "bottom": None}
                 for i in range(npatch)}

    new_indices = []
    new_patches = []
    for i in range(npatch):
        # pixels sur chaque bord de patch i
        left_i   = patch_indices[i][:, 0]
        right_i  = patch_indices[i][:, -1]
        top_i    = patch_indices[i][0, :]
        bottom_i = patch_indices[i][-1, :]

        # voisins HEALPix pour chaque bord
        neigh_left   = np.unique(hp.get_all_neighbours(nside_map, left_i))
        neigh_right  = np.unique(hp.get_all_neighbours(nside_map, right_i))
        neigh_top    = np.unique(hp.get_all_neighbours(nside_map, top_i))
        neigh_bottom = np.unique(hp.get_all_neighbours(nside_map, bottom_i))

        tac = []
        for j in range(npatch):
            if i == j:
                continue

            # pixels sur chaque bord du patch j
            left_j   = patch_indices[j][:, 0]
            right_j  = patch_indices[j][:, -1]
            top_j    = patch_indices[j][0, :]
            bottom_j = patch_indices[j][-1, :]
            
            new_neighs = [left_j, right_j, top_j, bottom_j]

            for k, new_neigh in enumerate(new_neighs):
                if set(new_neigh).issubset(neigh_left):
                    neighbors[i]["left"] = j
                    #tac.append(j)

                if set(new_neigh).issubset(neigh_right):
                    neighbors[i]["right"] = j
                    #tac.append(j)
            
                    if k == 0:
                        #right_patch = patches[j]
                        new_patch = np.hstack([patches[i], patches[j]])[:,int(w/2):-int(w/2)]
                        new_index = np.hstack([patch_indices[i], patch_indices[j]])[:,int(w/2):-int(w/2)]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    else:0
                    '''
                    elif k == 1:
                        #right_patch = patches[j][:,::-1]
                        new_patch = np.hstack([patches[i], patches[j][:,::-1]])[:,int(w/2):-int(w/2)]
                        new_index = np.hstack([patch_indices[i], patch_indices[j][:,::-1]])[:,int(w/2):-int(w/2)]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    else:0
                    elif k == 2:
                        #right_patch = patches[j].T[:,::-1]
                        new_patch = np.hstack([patches[i], patches[j].T[:,::-1]])[:,int(w/2):-int(w/2)]
                        new_index = np.hstack([patch_indices[i], patch_indices[j].T[:,::-1]])[:,int(w/2):-int(w/2)]
                        #print('do we have a problem here?')
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    else:
                        #right_patch = patches[j].T[::-1,:]
                        new_patch = np.hstack([patches[i], patches[j].T[::-1,:]])[:,int(w/2):-int(w/2)]
                        new_index = np.hstack([patch_indices[i], patch_indices[j].T[::-1,:]])[:,int(w/2):-int(w/2)]
                        #print('do we have a problem here?')
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    '''
                    #new_patches.append(new_patch)
                    #new_indices.append(new_index)
                    
                if set(new_neigh).issubset(neigh_bottom):
                    neighbors[i]["bottom"] = j
                    #tac.append(j)

                if set(new_neigh).issubset(neigh_top):
                    '''
                    if k == 0:
                        new_patch = np.vstack([patches[i], patches[j].T])[int(w/2):-int(w/2),:]
                        new_index = np.vstack([patch_indices[i], patch_indices[j].T])[int(w/2):-int(w/2),:]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    elif k == 1:
                        new_patch = np.vstack([patches[i], patches[j].T])[int(w/2):-int(w/2),:]
                        new_index = np.vstack([patch_indices[i], patch_indices[j].T])[int(w/2):-int(w/2),:]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    elif k == 2:
                        new_patch = np.vstack([patches[i], patches[j]])[int(w/2):-int(w/2),:]
                        new_index = np.vstack([patch_indices[i], patch_indices[j]])[int(w/2):-int(w/2),:]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    else:
                    '''
                    if k == 3:
                        new_patch = np.vstack([patches[i], patches[j]])[int(w/2):-int(w/2),:]
                        new_index = np.vstack([patch_indices[i], patch_indices[j]])[int(w/2):-int(w/2),:]
                        #new_patch = np.vstack([patches[i], patches[j][::-1,:]])[int(w/2):-int(w/2),:]
			#new_index = np.vstack([patch_indices[i], patch_indices[j][::-1,:]])[int(w/2):-int(w/2),:]
                        new_patches.append(new_patch)
                        new_indices.append(new_index)
                    #new_patches.append(new_patch)
                    #new_indices.append(new_index)
                    
                    neighbors[i]["top"] = j
                    #tac.append(j)
                    #neighbors.append(tac)


                    '''
                    In [219]: new_patches = []
     ...: new_indices = []
     ...: for i in range(len(neighbors)):
     ...:     right = neighbors[i]['right']
     ...:     if right:
     ...:         top_right = neighbors[right]['top']
     ...:         if top_right:
     ...:             left_top_right = neighbors[top_right]['left']
     ...:             if (left_top_right):
     ...:                 if neighbors[i]['top']:
     ...:                     if left_top_right == neighbors[i]['top']:
     ...:                         a = np.hstack([patches[neighbors[i]['top']], patches[top_right]])
     ...:                         b = np.hstack([patches[i], patches[right]])
     ...:                         c = np.vstack([a, b])[32:-32,32:-32]
     ...:                         aa = np.hstack([patch_indices[neighbors[i]['top']], patch_indices[top_right]])
     ...:                         bb = np.hstack([patch_indices[i], patch_indices[right]])
     ...:                         cc = np.vstack([aa, bb])[32:-32,32:-32]
     ...:                         new_patches.append(c)
     ...:                         new_indices.append(cc)
                    '''
                        
    return neighbors, np.array(new_patches), np.array(new_indices)
