### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$ = set of products, indexed by $p$ (from file_1_view_0.ProductName)
- $x_{sp}$ = number of units of product $p$ placed on shelf $s$ (decision variable, integer, $\geq 0$)
- $v_p$ = value of product $p$ (from file_1_view_0.Value)
- $w_p$ = weight of product $p$ (from file_1_view_0.Weight)
- $C_s$ = capacity of shelf $s$ (from file_0_view_0.Capacity)

#### Objective
Maximize total value of products allocated to all shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for each shelf $s$):
$$
\sum_{p \in P} w_p \, x_{sp} \leq C_s \quad \forall s \in S
$$

2. **Nonnegativity and Integrality** (for all shelves $s$ and products $p$):
$$
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

---

### Data Mapping

- $S$: All ShelfID in file_0_view_0.ShelfID (from capacity.csv)
- $P$: All ProductName in file_1_view_0.ProductName (from products.csv)
- $C_s$: file_0_view_0.Capacity, keyed by ShelfID
- $v_p$: file_1_view_0.Value, keyed by ProductName
- $w_p$: file_1_view_0.Weight, keyed by ProductName
- $x_{sp}$: Number of units of product $p$ on shelf $s$ (decision variable, integer, $\geq 0$)

All index sets, parameters, and constraints are defined directly from the supplied CSV data.