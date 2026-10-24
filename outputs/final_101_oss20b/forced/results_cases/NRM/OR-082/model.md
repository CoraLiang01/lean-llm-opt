**Symbolic Model**

```
Sets
    N  :={Depot, A, B, C}          # all locations
    C  :={A, B, C}                 # customer locations

Parameters
    d[i,j]  : distance between i and j,  i,j∈N, i≠j

Variables
    x[i,j]  : binary, 1 if the route goes directly from i to j
    u[i]    : integer, 1 ≤ u[i] ≤ |C|  for i∈C   (MTZ order variables)

Objective
    minimize   Σ_{i∈N} Σ_{j∈N, j≠i} d[i,j] · x[i,j]

Constraints
    1. Depot outgoing
          Σ_{j∈C} x[Depot,j] = 1

    2. Depot incoming
          Σ_{i∈C} x[i,Depot] = 1

    3. Customer outgoing
          ∀ i∈C : Σ_{j∈N, j≠i} x[i,j] = 1

    4. Customer incoming
          ∀ i∈C : Σ_{j∈N, j≠i} x[j,i] = 1

    5. MTZ subtour elimination
          ∀ i,j∈C, i≠j :
                u[i] - u[j] + |C| · x[i,j] ≤ |C| - 1
          (here |C| = 3, so  u[i] - u[j] + 3·x[i,j] ≤ 2)

    6. Order variable bounds
          ∀ i∈C : 1 ≤ u[i] ≤ |C|

    7. Binary variables
          ∀ i,j∈N, i≠j : x[i,j] ∈ {0,1}
```

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| `d[i,j]` | Distance matrix | `file_0_view_0` | `Depot`, `A`, `B`, `C` | `d[Depot,A]` = value in row *Depot*, column *A*; similarly for all ordered pairs (i,j) with i≠j. |
| `N` | Node list | `file_0_view_0` | `Unnamed: 0` | `N = {Depot, A, B, C}` extracted from the row identifiers. |
| `C` | Customer list | `file_0_view_0` | `Unnamed: 0` | `C = {A, B, C}` (all nodes except *Depot*). |

No numerical values are embedded in the model; all coefficients are linked to the source table `file_0_view_0` as indicated above.