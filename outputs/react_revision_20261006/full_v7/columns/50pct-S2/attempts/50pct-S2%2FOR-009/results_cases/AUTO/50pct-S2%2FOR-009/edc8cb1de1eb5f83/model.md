Mathematical Optimization Model

Index Sets:
- $I$: Set of areas available for development, indexed by $i$ (from ProductName in file_1_view_0).

Parameters:
- $v_i$: Development benefit of area $i$ (Value from file_1_view_0, column Value).
- $w_i$: Resource requirement (e.g., land, cost, or effort) for developing area $i$ (Weight from file_1_view_0, column Weight).
- $C$: Overall development capacity (Capacity from file_0_view_0, column Capacity).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\]

Data Mapping:
- $I$: All ProductName values in file_1_view_0 (products.csv), preserve source order.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are defined directly from the returned CSV data, with no omitted entities or constraints.