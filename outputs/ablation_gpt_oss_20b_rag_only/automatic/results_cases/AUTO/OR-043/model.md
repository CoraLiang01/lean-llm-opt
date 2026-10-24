**Mathematical Model**

```
SETS
    P   : set of products (ProductName)

PARAMETERS
    Value[p]   : benefit per unit of product p   (from file_1_view_0, column Value)
    Weight[p]  : weight (space) per unit of product p   (from file_1_view_0, column Weight)
    Capacity   : total stock capacity   (from file_0_view_0, column Capacity)

VARIABLES
    x[p]   : number of units of product p to order each day
             integer,  x[p] ≥ 0

OBJECTIVE
    maximize  Z = Σ_{p∈P}  Value[p] * x[p]

CONSTRAINTS
    CapacityLimit : Σ_{p∈P}  Weight[p] * x[p]  ≤  Capacity
```

**Data Mapping**

| Parameter | Source Table | Column | Index Key |
|-----------|--------------|--------|-----------|
| Value[p]  | file_1_view_0 | Value  | ProductName |
| Weight[p] | file_1_view_0 | Weight | ProductName |
| Capacity  | file_0_view_0 | Capacity | – |

*The set `P` is the collection of all `ProductName` values present in `file_1_view_0`.*