**Symbolic Model**

```
SETS
    I   : suppliers   (from fixed_cost.csv)
    J   : supermarkets (from demand.csv)

PARAMETERS
    f[i]   : fixed opening cost of supplier i          (i ∈ I)
    d[j]   : demand of supermarket j                   (j ∈ J)
    t[i,j] : transportation cost per unit from i to j (i ∈ I, j ∈ J)

VARIABLES
    y[i]   : 1 if supplier i is opened, 0 otherwise   (i ∈ I)
    x[i,j] : quantity shipped from i to j              (i ∈ I, j ∈ J)

BOUNDS
    0 ≤ x[i,j] ≤ d[j] * y[i]   ∀ i ∈ I, j ∈ J
    y[i] ∈ {0,1}               ∀ i ∈ I

CONSTRAINTS
    Demand satisfaction:
        ∑_{i∈I} x[i,j] = d[j]          ∀ j ∈ J

OBJECTIVE
    Minimize  ∑_{i∈I} f[i] * y[i]  +  ∑_{i∈I} ∑_{j∈J} t[i,j] * x[i,j]
```

**Data Mapping**

| Parameter | Table ID          | Row Key Column | Value Column |
|-----------|-------------------|----------------|--------------|
| `f[i]`    | `file_1_view_0`   | `Unnamed: 0`   | `fixed_costs` |
| `d[j]`    | `file_0_view_0`   | `customer`     | `demand` |
| `t[i,j]`  | `file_2_view_0`   | `Unnamed: 0`   | `C1`–`C25` (customer columns) |

*Notes*

- The set `I` is the collection of all supplier identifiers appearing in `fixed_cost.csv`.  
- The set `J` is the collection of all supermarket identifiers appearing in `demand.csv`.  
- The bound `x[i,j] ≤ d[j] * y[i]` enforces that a closed supplier cannot ship any goods; the right‑hand side uses the demand of supermarket `j` as a convenient big‑M.  
- All parameters are read directly from the specified tables; no numerical values are enumerated in the model.