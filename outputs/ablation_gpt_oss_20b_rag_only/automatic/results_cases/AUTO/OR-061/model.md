**Mathematical Model (Facility‑Location Formulation)**  

```
SETS
    I   {i}   : suppliers          (from file_1_view_0)
    J   {j}   : branches           (from file_0_view_0)

PARAMETERS
    f[i]          : fixed cost of opening supplier i
    c[i,j]        : transportation cost per unit from supplier i to branch j
    d[j]          : demand of branch j

VARIABLES
    y[i]          : 1 if supplier i is opened, 0 otherwise   (binary)
    x[i,j]        : quantity shipped from i to j              (continuous, ≥0)

CONSTRAINTS
    /* Demand must be met exactly */
    demand[j] :  ∑_{i∈I} x[i,j]  =  d[j]          ∀ j∈J

    /* Shipments only from open suppliers */
    link[i,j] :  x[i,j]  ≤  d[j] * y[i]          ∀ i∈I, j∈J

    /* Binary opening decisions */
    bin[i]    :  y[i]  ∈ {0,1}                    ∀ i∈I

OBJECTIVE
    minimize  ∑_{i∈I} f[i] * y[i]  +  ∑_{i∈I} ∑_{j∈J} c[i,j] * x[i,j]
```

---

**Data‑Mapping Rules**

| Parameter | Table ID          | Row‑ID Column | Column(s) Used | Notes |
|-----------|-------------------|---------------|----------------|-------|
| `f[i]`    | `file_1_view_0`   | `Unnamed: 0`  | `fixed_costs`  | Supplier identifier is the row‑ID. |
| `c[i,j]`  | `file_2_view_0`   | `Unnamed: 0`  | `C1`–`C5`      | Row‑ID gives supplier `i`; column names give branch `j`. |
| `d[j]`    | `file_0_view_0`   | `customer`    | `demand`       | Row‑ID gives branch `j`. |

All index sets `I` and `J` are taken from the full set of suppliers and branches present in the respective tables; no additional filtering is applied. The model uses the exact raw identifiers from the CSV files, preserving the original cardinalities (5 suppliers, 5 branches).