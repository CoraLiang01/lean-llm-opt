ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of products, indexed by $i$, corresponding to ProductName in file_1_view_0.

Parameters:
- $v_i$: value (benefit) per unit of product $i$ (from Value in file_1_view_0, key: ProductName).
- $w_i$: weight (stock space required) per unit of product $i$ (from Weight in file_1_view_0, key: ProductName).
- $C$: overall stock capacity (from Capacity in file_0_view_0).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ to order each day.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \text{and integer}, \quad \forall i \in I
\]

DATA MAPPING

- $I$: All ProductName in file_1_view_0.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are defined directly from the current CSV data.