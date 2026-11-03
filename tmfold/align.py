"""Map structure residues onto query-sequence positions.

Fast paths cover identical sequences and exact substrings (AlphaFold DB fragments, trimmed
constructs). Otherwise a semi-global alignment (free end gaps) is used, which tolerates
expression tags, missing loops and point mutations in experimental structures.
"""
from __future__ import annotations

import numpy as np

MATCH, MISMATCH, GAP = 2.0, -1.0, -2.0


def map_positions(struct_seq: str, query: str) -> dict[int, int]:
    """Return {structure index (0-based): query position (1-based)} for aligned residues."""
    if struct_seq == query:
        return {i: i + 1 for i in range(len(query))}
    off = query.find(struct_seq)
    if struct_seq and off >= 0:
        return {i: off + i + 1 for i in range(len(struct_seq))}
    return _semi_global(struct_seq, query)


def _semi_global(a: str, b: str) -> dict[int, int]:
    """Semi-global alignment of a (structure) to b (query); end gaps are free in both."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return {}
    A = np.frombuffer(a.encode(), np.uint8)
    B = np.frombuffer(b.encode(), np.uint8)
    H = np.zeros((n + 1, m + 1))
    ptr = np.zeros((n + 1, m + 1), np.int8)   # 0 diag, 1 up (gap in b), 2 left (gap in a)
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2
    j_idx = np.arange(1, m + 1)
    for i in range(1, n + 1):
        s = np.where(B == A[i - 1], MATCH, MISMATCH)
        diag = H[i - 1, :-1] + s
        up = H[i - 1, 1:] + GAP
        T = np.maximum(diag, up)
        tptr = np.where(diag >= up, 0, 1).astype(np.int8)
        # left moves: H[j] = max_k<=j (T[k] + GAP*(j-k)), with H[i,0] = 0 (free leading gap)
        base = np.concatenate([[H[i, 0]], T]) - GAP * np.arange(0, m + 1)
        best = np.maximum.accumulate(base)[1:] + GAP * j_idx
        row = np.maximum(T, best)
        H[i, 1:] = row
        ptr[i, 1:] = np.where(row > T, 2, tptr)
    # free trailing gaps: best end in last row or last column
    i_end, j_end = n, int(np.argmax(H[n]))
    i_alt = int(np.argmax(H[:, m]))
    if H[i_alt, m] > H[n, j_end]:
        i_end, j_end = i_alt, m
    mapping, i, j = {}, i_end, j_end
    while i > 0 and j > 0:
        p = ptr[i, j]
        if p == 0:
            mapping[i - 1] = j  # matches and substitutions both map (point mutants keep their position)
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    return mapping
