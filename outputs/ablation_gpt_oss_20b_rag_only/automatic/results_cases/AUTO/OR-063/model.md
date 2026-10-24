**Symbolic Model (Facility‑Location Problem)**  

```
Sets
    I   : {S1,S2,S3,S4,S5,S6,S7}          // warehouses
    J   : {C1,C2,C3,C4,C5,C6,C7}          // customers

Parameters
    f[i]   : fixed cost of opening warehouse i
    c[i,j] : transportation cost per unit from warehouse i to customer j
    d[j]   : demand of customer j

Decision Variables
    y[i]   ∈ {0,1}   // 1 if warehouse i is opened
    x[i,j] ≥ 0       // quantity shipped from i to j

Objective
    minimize   Σ_{i∈I} f[i]·y[i]  +  Σ_{i∈I} Σ_{j∈J} c[i,j]·x[i,j]

Constraints
    // Demand satisfaction
    ∀ j∈J :  Σ_{i∈I} x[i,j]  ≥  d[j]

    // Shipments only from opened warehouses
    ∀ i∈I, j∈J :  x[i,j]  ≤  d[j] · y[i]          // (big‑M = d[j])

    // Variable domains
    ∀ i∈I :  y[i] ∈ {0,1}
    ∀ i∈I, j∈J :  x[i,j] ≥ 0
```

---

**Exact Data‑Mapping Rules**

| Parameter | Source Table | Row‑ID Column | Value Column | Mapping |
|-----------|--------------|---------------|--------------|---------|
| `f[i]`    | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | `i = Unnamed: 0` |
| `d[j]`    | `file_0_view_0` | `customer`   | `demand`      | `j = customer` |
| `c[i,j]`  | `file_2_view_0` | `Unnamed: 0` | `C1`–`C7`     | `i = Unnamed: 0`, `j = column‑name` |

*All indices `i` and `j` are taken from the full set of rows/columns returned in the observation; no additional filtering is applied.*