**Symbolic Model**

```
Sets
    I   : Display areas          (indices = DisplayID from capacity.csv)
    J   : Boat types             (indices = ProductName from products.csv)

Parameters
    Capacity[i]   : capacity of display area i          (from capacity.csv)
    Value[j]      : monetary value of boat type j       (from products.csv)
    Weight[j]     : physical size (weight) of boat type j (from products.csv)

Decision Variables
    x[i,j]  : number of boats of type j placed in display area i
              integer,  x[i,j] ≥ 0

Objective
    maximize  Z = Σ_{i∈I} Σ_{j∈J}  Value[j] * x[i,j]

Constraints
    (Capacity limits)
    Σ_{j∈J}  Weight[j] * x[i,j]  ≤  Capacity[i]      ∀ i ∈ I

    (Non‑negativity)
    x[i,j]  ≥ 0  ∀ i∈I, j∈J
```

**Data Mapping**

| Parameter | Table ID          | Column Name |
|-----------|-------------------|-------------|
| Capacity  | file_0_view_0     | Capacity    |
| Value     | file_1_view_0     | Value       |
| Weight    | file_1_view_0     | Weight      |

All indices (`I`, `J`) are taken directly from the rows of the corresponding tables; no additional filtering or enumeration is applied. The model uses only the data supplied by the two CSV files and follows the objective and constraints stated in the query.