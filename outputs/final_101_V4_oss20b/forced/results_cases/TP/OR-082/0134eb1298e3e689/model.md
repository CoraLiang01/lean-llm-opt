**Symbolic Model**

```
Sets
    N  : {Depot, A, B, C}
    V  : N \ {Depot}          # {A, B, C}

Parameters
    d[i,j]  : distance from i to j   (i,j ∈ N, i≠j)

Decision Variables
    x[i,j]  ∈ {0,1}   for all i,j ∈ N, i≠j
    u[i]    ∈ [1,|V|] for all i ∈ V

Objective
    minimize   ∑_{i∈N} ∑_{j∈N, j≠i} d[i,j] · x[i,j]

Constraints
    1. Out‑degree  :  ∑_{j∈N, j≠i} x[i,j] = 1          ∀ i∈N
    2. In‑degree   :  ∑_{i∈N, i≠j} x[i,j] = 1          ∀ j∈N
    3. MTZ Subtour elimination :
           u[i] – u[j] + |V| · x[i,j] ≤ |V| – 1      ∀ i∈V, j∈V, i≠j
```

---

**Data‑Mapping Rules**

| Element | Source | Mapping Details |
|---------|--------|-----------------|
| `N` | `file_0_view_0` | Row/column identifiers: `Depot`, `A`, `B`, `C` |
| `V` | derived | `N` minus `Depot` |
| `d[i,j]` | `file_0_view_0` | For each pair (i,j) with i≠j, read the cell value in the row whose `Unnamed: 0` equals i and the column whose header equals j. |
| `x[i,j]` | decision variable | Binary variable for each ordered pair (i,j), i≠j |
| `u[i]` | decision variable | Continuous variable for each i∈V, domain [1,3] |
| Objective coefficient | `d[i,j]` | As above |
| Constraint 1 & 2 | `x[i,j]` | Sum over outgoing/incoming arcs |
| Constraint 3 | `x[i,j]`, `u[i]`, `u[j]` | MTZ inequality |

**Table Identification**

- **Table ID**: `file_0_view_0`
- **Columns**: `Unnamed: 0`, `Depot`, `A`, `B`, `C`
- **Row identifiers**: values of `Unnamed: 0` (`Depot`, `A`, `B`, `C`)

**Parameter Extraction Example**

| i | j | d[i,j] |
|---|---|--------|
| Depot | A | 28 |
| Depot | B | 41 |
| Depot | C | 63 |
| A | Depot | 28 |
| A | B | 27 |
| A | C | 87 |
| B | Depot | 41 |
| B | A | 27 |
| B | C | 81 |
| C | Depot | 63 |
| C | A | 87 |
| C | B | 81 |

(Values are taken directly from the matrix cells; all other entries are omitted for brevity.)

The model above fully specifies the traveling‑salesman problem for the depot and three customers, using the distance matrix supplied in `DistanceMatrix.csv`.