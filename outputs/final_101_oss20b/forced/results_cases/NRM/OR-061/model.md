**Symbolic Model**

```
Sets
    I   : suppliers
    J   : customers

Parameters
    f[i]   : fixed cost of opening supplier i
    d[j]   : demand of customer j
    c[i,j] : transportation cost per unit from supplier i to customer j

Variables
    y[i]   : 1 if supplier i is opened, 0 otherwise          (binary)
    x[i,j] : quantity shipped from supplier i to customer j  (continuous, ≥0)

Objective
    minimize  Z = Σ_{i∈I} f[i] y[i]  +  Σ_{i∈I} Σ_{j∈J} c[i,j] x[i,j]

Constraints
    1. Demand satisfaction
          Σ_{i∈I} x[i,j]  ≥  d[j]          ∀ j∈J

    2. Supplier activation (link x and y)
          x[i,j]  ≤  d[j] · y[i]           ∀ i∈I, j∈J

    3. Non‑negativity
          x[i,j]  ≥  0                     ∀ i∈I, j∈J

    4. Binary
          y[i]   ∈ {0,1}                   ∀ i∈I
```

---

**Data Mapping**

| Parameter / Index | Source Table | Source Column(s) | Index Column |
|-------------------|--------------|------------------|--------------|
| `I` (suppliers)   | `file_1_view_0` | `Unnamed: 0` | – |
| `J` (customers)   | `file_0_view_0` | `customer` | – |
| `f[i]`            | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` |
| `d[j]`            | `file_0_view_0` | `demand` | `customer` |
| `c[i,j]`          | `file_2_view_0` | `C1`–`C5` (columns for each customer) | `Unnamed: 0` (rows) and customer columns |

*All numerical values are taken directly from the corresponding CSV files; the model uses only the symbolic representation above.*