**Symbolic Model**

```
Sets
    N  : = {Depot, A, B, C}          # all locations
    V  : = N \ {Depot}                # customers only

Parameters
    d[i,j]  : distance from i to j,  i,j ∈ N, i ≠ j

Decision Variables
    x[i,j]  ∈ {0,1}   for all i,j ∈ N, i ≠ j   # 1 if the route goes directly from i to j
    u[i]    ∈ ℤ      for all i ∈ V             # position of customer i in the tour

Objective
    minimize   Σ_{i∈N} Σ_{j∈N, j≠i}  d[i,j] · x[i,j]

Constraints
1.  Out‑degree 1 for every node
        Σ_{j∈N, j≠i} x[i,j] = 1          ∀ i ∈ N

2.  In‑degree 1 for every node
        Σ_{i∈N, i≠j} x[i,j] = 1          ∀ j ∈ N

3.  MTZ subtour elimination (|V| = 3)
        u[i] – u[j] + |V| · x[i,j] ≤ |V| – 1
        ∀ i,j ∈ V,  i ≠ j

4.  Position bounds
        1 ≤ u[i] ≤ |V|                    ∀ i ∈ V
```

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| `d[i,j]`  | `file_0_view_0` (DistanceMatrix.csv) | `d[i,j]` = value in the cell whose row label is `i` and column label is `j`.  Row and column labels are taken from the first column (`Unnamed: 0`) and the header row (`Depot, A, B, C`). |
| `N`       | `file_0_view_0` | `N` = set of all unique labels appearing in the first column and header row. |
| `V`       | derived | `V` = `N` minus the label `"Depot"`. |

No numerical values are listed; the model references the data through the mapping rules above.