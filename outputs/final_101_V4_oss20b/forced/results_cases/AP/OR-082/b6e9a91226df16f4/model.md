**Mathematical Model**

```
Sets
  N  = {Depot, A, B, C}
  C  = {A, B, C}

Parameters
  d[i,j]   : distance between i and j   (i,j ∈ N)

Variables
  x[i,j]   ∈ {0,1}   for all i,j ∈ N, i ≠ j
  u[i]     ∈ {1,2,3} for all i ∈ C

Objective
  minimize   Σ_{i∈N} Σ_{j∈N, j≠i} d[i,j] * x[i,j]

Constraints
  1. Outgoing – each node has exactly one outgoing arc
     Σ_{j∈N, j≠i} x[i,j] = 1          ∀ i ∈ N

  2. Incoming – each node has exactly one incoming arc
     Σ_{i∈N, i≠j} x[i,j] = 1          ∀ j ∈ N

  3. MTZ subtour elimination (customers only)
     u[i] - u[j] + 4 * x[i,j] ≤ 3      ∀ i,j ∈ C, i ≠ j

  4. Rank bounds
     1 ≤ u[i] ≤ 3                     ∀ i ∈ C
```

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| `d[i,j]`  | `file_0_view_0` | `d[i,j] = value in row where Unnamed:0 = i and column j` |
| `x[i,j]`  | — | binary variable |
| `u[i]`    | — | integer variable in {1,2,3} |

**Explicit distance values (from `DistanceMatrix.csv`)**

```
d[Depot,A] = 28
d[Depot,B] = 41
d[Depot,C] = 63
d[A,Depot] = 28
d[A,B]     = 27
d[A,C]     = 87
d[B,Depot] = 41
d[B,A]     = 27
d[B,C]     = 81
d[C,Depot] = 63
d[C,A]     = 87
d[C,B]     = 81
d[Depot,Depot] = 0
d[A,A] = 0
d[B,B] = 0
d[C,C] = 0
```

All other `d[i,j]` not listed are implicitly zero (diagonal) or follow the same mapping rule.