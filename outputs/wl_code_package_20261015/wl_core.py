"""Cumulative WL subtree feature counts and cosine-normalized kernels.

For each round i, the shared dictionary maps
(old label, sorted multiset of neighbor old labels) to a new label.
Coordinates use (round, label), so features from different rounds never mix.
Graph encoding and LP normalization are in run_wl.py.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


def wl_kernels(graphs, max_h=3):
    """Cosine-normalized cumulative WL subtree kernels for h=0,...,max_h.

    All graphs share a signature dictionary each round. At h, the feature vector
    contains label counts from rounds 0 through h, matching the cumulative WL
    subtree construction.
    """
    labels = [label[:] for label, _ in graphs]
    accumulated, kernels = [Counter() for _ in graphs], {}
    for h in range(max_h + 1):
        for index, current in enumerate(labels): accumulated[index].update((h, label) for label in current)
        vocab, rr, cc, values = {}, [], [], []
        for graph_index, counts in enumerate(accumulated):
            for key, count in counts.items():
                vocab.setdefault(key, len(vocab)); rr.append(graph_index); cc.append(vocab[key]); values.append(count)
        features = csr_matrix((np.asarray(values, dtype=float), (rr, cc)), shape=(len(graphs), len(vocab)))
        raw = (features @ features.T).toarray(); norm = np.sqrt(np.diag(raw))
        if np.any(norm == 0): raise ValueError("Cannot normalize an empty graph")
        kernels[h] = np.clip(raw / np.outer(norm, norm), 0.0, 1.0)
        if h == max_h: continue
        codebook, updated_all = {}, []
        for current, (_, adjacency) in zip(labels, graphs):
            updated = []
            for vertex, neighbors in enumerate(adjacency):
                signature = (current[vertex], tuple(sorted(current[x] for x in neighbors)))
                codebook.setdefault(signature, len(codebook)); updated.append(codebook[signature])
            updated_all.append(updated)
        labels = updated_all
    return kernels


def self_test():
    """Check invariance to node reordering and disjoint graph replication."""
    graph = (["R:I", "V:C", "V:C"], [[1, 2], [0], [0]])
    reordered = (["V:C", "R:I", "V:C"], [[1], [0, 2], [1]])
    repeated = (graph[0] * 2, [[v + k * 3 for v in neighbor] for k in range(2) for neighbor in graph[1]])
    for matrix in wl_kernels([graph, reordered, repeated], 3).values():
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1.0)
        assert np.isclose(matrix[0, 1], 1.0) and np.isclose(matrix[0, 2], 1.0)
    return {"permutation_invariant": True, "disjoint_replication_similarity": 1.0}
