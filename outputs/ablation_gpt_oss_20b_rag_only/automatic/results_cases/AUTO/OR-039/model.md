**Symbolic Model**

```
Sets
    I   – product types
    W   – warehouses

Parameters
    v[i]   – value per unit of product i          (i ∈ I)
    c[w]   – capacity of warehouse w              (w ∈ W)

Decision Variables
    x[i,w]  – number of units of product i stored in warehouse w  (i ∈ I, w ∈ W)
              x[i,w] ∈ ℤ₊

Objective
    maximize   Z = Σ_{i∈I} Σ_{w∈W} v[i] · x[i,w]

Constraints
    (Capacity)   ∀ w ∈ W :   Σ_{i∈I} x[i,w]  ≤  c[w]
```

**Data Mapping**

| Symbol | Table ID          | Column Name   | Indexing |
|--------|-------------------|---------------|----------|
| I      | file_1_view_0     | ProductName   | set of distinct values |
| W      | file_0_view_0     | Warehouse ID  | set of distinct values |
| v[i]   | file_1_view_0     | Value         | indexed by ProductName |
| c[w]   | file_0_view_0     | Capacity      | indexed by Warehouse ID |

*All other columns (e.g., “Weight”) are not used in the model.*