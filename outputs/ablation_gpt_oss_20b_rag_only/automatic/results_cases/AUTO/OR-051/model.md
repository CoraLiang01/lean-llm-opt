**Mathematical Model**

```
SETS
    I   : set of cabinets
    J   : set of coffee products

PARAMETERS
    Capacity[i]   : weight capacity of cabinet i   (i ∈ I)
    Value[j]      : monetary value per unit of product j   (j ∈ J)
    Weight[j]     : weight per unit of product j   (j ∈ J)

VARIABLES
    x[i,j]   : integer number of units of product j placed in cabinet i   (i ∈ I, j ∈ J)
    x[i,j] ≥ 0, integer

OBJECTIVE
    maximize  Z = Σ_{i∈I} Σ_{j∈J} Value[j] * x[i,j]

CONSTRAINTS
    CapacityLimit(i) : Σ_{j∈J} Weight[j] * x[i,j] ≤ Capacity[i]   ∀ i ∈ I
```

---

**Data Mapping**

| Parameter / Set | Table ID          | Key Column | Value Column |
|-----------------|-------------------|------------|--------------|
| I (CabinetID)   | file_0_view_0     | CabinetID | –            |
| Capacity[i]     | file_0_view_0     | CabinetID | Capacity     |
| J (ProductName) | file_1_view_0     | ProductName | –          |
| Value[j]        | file_1_view_0     | ProductName | Value      |
| Weight[j]       | file_1_view_0     | ProductName | Weight     |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.