Mathematical Model

Index Sets:
- Let $I$ be the set of products, with each $i \in I$ corresponding to a row in file_1_view_0 (products.csv).

Parameters:
- $v_i$: Value (profit or benefit per unit) of product $i$ (from file_1_view_0, column Value).
- $w_i$: Weight (space or stock usage per unit) of product $i$ (from file_1_view_0, column Weight).
- $C$: Overall stock capacity (from file_0_view_0, column Capacity).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

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

Data Mapping

- $I$: All rows in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0, column Value, for each $i$.
- $w_i$: file_1_view_0, column Weight, for each $i$.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.