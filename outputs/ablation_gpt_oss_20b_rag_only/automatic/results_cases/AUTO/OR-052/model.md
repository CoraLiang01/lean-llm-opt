**Mathematical Model**

```
SETS
    I   : = {BookshelfID}          // all rows of capacity.csv
    J   : = {ProductName}          // all rows of products.csv

PARAMETERS
    Capacity[i]   : capacity of bookshelf i          (i ∈ I)
    Value[j]      : value of book j                  (j ∈ J)
    Weight[j]     : weight of book j                 (j ∈ J)

VARIABLES
    x[i,j]        : integer number of units of book j placed on bookshelf i
                    (i ∈ I, j ∈ J)

OBJECTIVE
    maximize  Z = Σ_{i∈I} Σ_{j∈J} Value[j] * x[i,j]

CONSTRAINTS
    // Weight capacity of each bookshelf
    ∀ i ∈ I :  Σ_{j∈J} Weight[j] * x[i,j]  ≤  Capacity[i]

    // Non‑negativity and integrality
    ∀ i ∈ I, j ∈ J :  x[i,j]  ∈  ℤ_+   (integer, ≥ 0)
```

---

**Data Mapping**

| Parameter / Index | Table ID          | Key Column(s) | Value Column |
|-------------------|-------------------|---------------|--------------|
| `I` (BookshelfID) | `file_0_view_0`   | `BookshelfID` | –            |
| `Capacity[i]`     | `file_0_view_0`   | `BookshelfID` | `Capacity`   |
| `J` (ProductName) | `file_1_view_0`   | `ProductName` | –            |
| `Value[j]`        | `file_1_view_0`   | `ProductName` | `Value`      |
| `Weight[j]`       | `file_1_view_0`   | `ProductName` | `Weight`     |

All numeric columns are treated as real parameters; the decision variables `x[i,j]` are explicitly declared integer and non‑negative. No additional constraints or data are introduced beyond what is specified in the query and the supplied tables.