#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all ProductName in file_1_view_0)
- For each $i \in I$:
    - $v_i$ = Value of vehicle type $i$ (from Value in file_1_view_0)
    - $w_i$ = Weight of vehicle type $i$ (from Weight in file_1_view_0)
- $C$ = total inventory capacity (from Capacity in file_0_view_0)
- $x_i$ = number of units of vehicle type $i$ to order daily (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)