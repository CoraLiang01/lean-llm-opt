#### Abstract Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from ProductName in products.csv)
- $x_i$ = number of units of bread type $i$ to order each day (decision variable, integer, $x_i \geq 0$)
- $p_i$ = expected profit per unit of bread type $i$ (from Value in products.csv)
- $w_i$ = storage weight per unit of bread type $i$ (from Weight in products.csv)
- $C$ = total storage capacity (from Capacity in capacity.csv)

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Subject to:**
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv), in source order.
- $p_i$: Value column in file_1_view_0 (products.csv), indexed by ProductName.
- $w_i$: Weight column in file_1_view_0 (products.csv), indexed by ProductName.
- $C$: Capacity column in file_0_view_0 (capacity.csv).

Each $x_i$ is the integer number of units of bread type $i$ to order each day. The total storage used cannot exceed $C$. The objective is to maximize total expected profit.