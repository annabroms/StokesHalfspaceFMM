"""Immutable occupied bins for the parallel FMM exclusion-radius check."""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _cell_hash(x, y, z, mask):
    """Mix three signed cell coordinates into a power-of-two hash table."""
    value = (np.uint64(x) * np.uint64(73856093)
             ^ np.uint64(y) * np.uint64(19349663)
             ^ np.uint64(z) * np.uint64(83492791))
    value ^= value >> np.uint64(33)
    value *= np.uint64(0xff51afd7ed558ccd)
    value ^= value >> np.uint64(33)
    return np.int64(value & np.uint64(mask))


@njit(cache=True)
def _build_bins(cells):
    """Build occupied-cell heads and source chains in O(N) storage without Python objects."""
    capacity = 1
    while capacity < 2 * len(cells):
        capacity *= 2
    keys = np.full(capacity, -1, dtype=np.int64)
    heads = np.full(capacity, -1, dtype=np.int64)
    following = np.full(len(cells), -1, dtype=np.int64)
    mask = capacity - 1
    for i in range(len(cells)):
        x, y, z = cells[i, 0], cells[i, 1], cells[i, 2]
        slot = _cell_hash(x, y, z, mask)
        while keys[slot] >= 0:
            entry = keys[slot]
            if cells[entry, 0] == x and cells[entry, 1] == y and cells[entry, 2] == z:
                break
            slot = (slot + 1) & mask
        if keys[slot] < 0:
            keys[slot] = i
        following[i] = heads[slot]
        heads[slot] = i
    return keys, heads, following


@njit(parallel=True, fastmath=False, cache=True)
def _query_bins(source, target, source_cells, target_cells, keys, heads, following, tau):
    """Search 27 adjacent cells independently per target using the original inclusive norm test."""
    hits = np.full(len(target), -1, dtype=np.int64)
    mask = len(keys) - 1
    for i in prange(len(target)):
        found = -1
        cx, cy, cz = target_cells[i, 0], target_cells[i, 1], target_cells[i, 2]
        for dx in range(-1, 2):
            if found >= 0:
                break
            for dy in range(-1, 2):
                if found >= 0:
                    break
                for dz in range(-1, 2):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    slot = _cell_hash(x, y, z, mask)
                    entry = keys[slot]
                    while entry >= 0:
                        if (source_cells[entry, 0] == x and source_cells[entry, 1] == y
                                and source_cells[entry, 2] == z):
                            candidate = heads[slot]
                            while candidate >= 0:
                                a = (source[candidate, 0] - target[i, 0]) / tau
                                b = (source[candidate, 1] - target[i, 1]) / tau
                                c = (source[candidate, 2] - target[i, 2]) / tau
                                if a*a + b*b + c*c <= 1.0:
                                    found = candidate
                                    break
                                candidate = following[candidate]
                            break
                        slot = (slot + 1) & mask
                        entry = keys[slot]
                    if found >= 0:
                        break
        hits[i] = found
    return hits


def unresolved_pairs(sources=None, targets=None, relative_threshold=64*np.finfo(float).eps):
    """Return an offending source per target; no arguments run a deterministic self-test."""
    if sources is None and targets is None:
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        probes = np.array([[relative_threshold, 0., 0.], [0.5, 0.5, 0.5]])
        hits = unresolved_pairs(points, probes, relative_threshold)
        np.testing.assert_array_equal(hits, [0, -1])
        return None
    if not len(sources) or not len(targets):
        return np.full(len(targets), -1, dtype=np.int64)
    origin = np.minimum(sources.min(axis=0), targets.min(axis=0))
    cell_width = 2*relative_threshold
    source_cells = np.floor((sources-origin)/cell_width).astype(np.int64)
    target_cells = np.floor((targets-origin)/cell_width).astype(np.int64)
    keys, heads, following = _build_bins(source_cells)
    return _query_bins(sources, targets, source_cells, target_cells,
                       keys, heads, following, relative_threshold)
