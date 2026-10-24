**Mathematical Model**

```
Indices
    P  : set of product names  (from file_1_view_0, column ProductName)

Parameters
    Profit[p]   : expected profit per unit of product p   (file_1_view_0, column Value)
    Weight[p]   : storage space per unit of product p    (file_1_view_0, column Weight)
    Capacity    : total available storage capacity       (file_0_view_0, column Capacity)

Variables
    x[p]        : number of units of product p to order each day
                  x[p] ∈ ℤ,  x[p] ≥ 0

Objective
    maximize  Z = Σ_{p∈P} Profit[p] · x[p]

Constraints
    Storage: Σ_{p∈P} Weight[p] · x[p]  ≤  Capacity
```

---

**Data Mapping**

| Symbol | Type | Source Table | Source Column |
|--------|------|--------------|---------------|
| P      | index set | file_1_view_0 | ProductName |
| Profit[p] | parameter | file_1_view_0 | Value |
| Weight[p] | parameter | file_1_view_0 | Weight |
| Capacity | parameter | file_0_view_0 | Capacity |
| x[p] | variable | – | – |
| Domain of x[p] | integer, lower bound 0 | – | – |

*Objective sense:* maximize  
*Constraint sense:* ≤
