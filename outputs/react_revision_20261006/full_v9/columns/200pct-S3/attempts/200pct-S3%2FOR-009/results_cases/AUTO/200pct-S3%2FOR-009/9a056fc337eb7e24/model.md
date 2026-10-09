ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas available for development (from file_1_view_0, column ProductName).

Parameters:
- $v_i$: Development benefit per unit scale in area $i$ (from file_1_view_0, column Value, key ProductName).
- $w_i$: Resource requirement per unit scale in area $i$ (from file_1_view_0, column Weight, key ProductName).
- $C$: Overall development capacity (from file_0_view_0, column Capacity).

Decision Variables:
- $x_i \geq 0$, integer: Scale of development per day in area $i$.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

DATA MAPPING

- $I$: All ProductName values in file_1_view_0 (products.csv).
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are defined directly from the supplied CSV data.