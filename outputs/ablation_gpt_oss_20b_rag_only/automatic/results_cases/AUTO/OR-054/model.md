**Mathematical Model**

```
Sets
    I   : set of shelves          (indexed by ShelfID)
    J   : set of products         (indexed by ProductName)

Parameters
    Capacity[i]   : capacity of shelf i          (from file_0_view_0, column Capacity)
    Value[j]      : value per unit of product j  (from file_1_view_0, column Value)
    Weight[j]     : weight per unit of product j (from file_1_view_0, column Weight)

Decision Variables
    x[i,j]  : number of units of product j placed on shelf i
              integer,  x[i,j] ≥ 0

Objective
    maximize  Z = Σ_{i∈I} Σ_{j∈J} Value[j] * x[i,j]

Constraints
    (Shelf capacity)   ∀ i∈I :
        Σ_{j∈J} Weight[j] * x[i,j]  ≤  Capacity[i]

    (Non‑negativity)   ∀ i∈I, j∈J :
        x[i,j]  ≥ 0   and  integer
```

---

**Data Mapping**

| Parameter / Index | Table ID          | Column(s) used | Index column |
|-------------------|-------------------|----------------|--------------|
| I (shelves)      | file_0_view_0     | ShelfID        | ShelfID      |
| Capacity[i]      | file_0_view_0     | Capacity       | ShelfID      |
| J (products)     | file_1_view_0     | ProductName    | ProductName  |
| Value[j]         | file_1_view_0     | Value          | ProductName  |
| Weight[j]        | file_1_view_0     | Weight         | ProductName  |

*All other columns in the source tables are ignored.*