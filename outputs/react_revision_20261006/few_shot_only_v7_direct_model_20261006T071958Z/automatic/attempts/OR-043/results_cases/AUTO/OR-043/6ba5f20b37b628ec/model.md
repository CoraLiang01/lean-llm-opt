ABSTRACT MATHEMATICAL MODEL

Sets:
- Let $I$ be the set of all drug products, indexed by $i$.

Parameters:
- $v_i$: Value (benefit) of one unit of product $i$.
- $w_i$: Weight (stock space required) of one unit of product $i$.
- $C$: Total stock capacity.

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

DATA MAPPING

- $I$: All records in table_id file_1_view_0, column ProductName.
- $v_i$: file_1_view_0, column Value, for each $i$.
- $w_i$: file_1_view_0, column Weight, for each $i$.
- $C$: file_0_view_0, column Capacity (single record).
- $x_i$: Decision variable for each $i \in I$.