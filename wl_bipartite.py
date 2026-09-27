"""Typed variable--constraint bipartite Weisfeiler--Lehman similarity for LP/MIP.

This implements the WL subtree feature construction of Shervashidze et al.
(2011) for the *chosen* LP representation. It is not an implementation of the
MIPLIB 2017 105-feature pipeline.
"""
from collections import Counter

import gurobipy as gp
import numpy as np
from scipy.sparse import csr_matrix


def read_linear_model(path, env):
    """Read a linear LP/MIP and return information used by this diagnostic."""
    model = gp.read(str(path), env=env)
    if model.NumQConstrs or model.NumGenConstrs or model.NumSOS or model.NumQNZs:
        raise ValueError(f"Unsupported quadratic, general, or SOS content: {path}")
    matrix = model.getA().tocsr(); matrix.eliminate_zeros()
    variables, rows = model.getVars(), model.getConstrs()
    output = {"A": matrix, "types": np.asarray(model.getAttr("VType", variables)),
              "senses": np.asarray(model.getAttr("Sense", rows)),
              "n": matrix.shape[1], "m": matrix.shape[0], "nnz": matrix.nnz}
    model.dispose()
    return output


def typed_graph(model):
    """Use nonzero A[i,j] edges, variable B/I/C and row equality/inequality labels.

    Objective, coefficient values/signs, RHS, bounds, and names do not enter.
    Both <= and >= rows receive R:I; equality receives R:E.
    """
    n, m, matrix = model["n"], model["m"], model["A"]
    if not set(model["types"]).issubset({"B", "I", "C"}): raise ValueError("Unexpected variable domain")
    if not set(model["senses"]).issubset({"<", "=", ">"}): raise ValueError("Unexpected row sense")
    labels = [f"V:{t}" for t in model["types"]] + ["R:E" if s == "=" else "R:I" for s in model["senses"]]
    adjacency = [[] for _ in range(n + m)]
    for row in range(m):
        start, stop = matrix.indptr[row:row + 2]
        for variable in matrix.indices[start:stop]:
            adjacency[variable].append(n + row); adjacency[n + row].append(int(variable))
    return labels, adjacency


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
