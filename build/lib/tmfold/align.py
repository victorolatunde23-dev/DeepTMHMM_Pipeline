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
    """Local (Smith–Waterman) alignment of a (structure) to b (query).

    Both an expression tag and the unmodelled part of the query can be skipped at no cost;
    point mutations and internal gaps are tolerated. (Named for backwards compatibility: a
    semi-global alignment cannot skip a tag on a construct that starts mid-sequence.)
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return {}
    A = np.frombuffer(a.encode(), np.uint8)
    B = np.frombuffer(b.encode(), np.uint8)
    H = np.zeros((n + 1, m + 1))
    ptr = np.full((n + 1, m + 1), 3, np.int8)   # 0 diag, 1 up (gap in b), 2 left (gap in a), 3 start
    j_idx = np.arange(1, m + 1)
    for i in range(1, n + 1):
        diag = H[i - 1, :-1] + np.where(B == A[i - 1], MATCH, MISMATCH)
        up = H[i - 1, 1:] + GAP
        T = np.maximum(np.maximum(diag, up), 0.0)
        tptr = np.where(T <= 0, 3, np.where(diag >= up, 0, 1)).astype(np.int8)
        base = np.concatenate([[0.0], T]) - GAP * np.arange(0, m + 1)
        row = np.maximum(T, np.maximum.accumulate(base)[1:] + GAP * j_idx)
        H[i, 1:] = row
        ptr[i, 1:] = np.where(row > T, 2, tptr)
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    mapping = {}
    while i > 0 and j > 0 and H[i, j] > 0:
        p = ptr[i, j]
        if p == 0:
            mapping[int(i) - 1] = int(j)  # matches and substitutions both map (point mutants keep their position)
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        elif p == 2:
            j -= 1
        else:
            break
    return mapping
