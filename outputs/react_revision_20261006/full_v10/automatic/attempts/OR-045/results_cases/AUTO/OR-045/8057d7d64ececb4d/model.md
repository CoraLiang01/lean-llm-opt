Mathematical Model

Sets:
- Let $I$ be the set of produce types, indexed by $i$ (from file_1_view_0, column ProductName).

Parameters:
- $v_i$: Value per unit of produce $i$ (file_1_view_0, column Value, key ProductName).
- $w_i$: Weight per unit of produce $i$ (file_1_view_0, column Weight, key ProductName).
- $C$: Total inventory capacity (file_0_view_0, column Capacity).

Decision Variables:
- $x_i$: Number of units of produce $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$.

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

Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv), preserve source order.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative).