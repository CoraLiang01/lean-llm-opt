Mathematical Optimization Model

Index Sets:
- $I$: Set of drug products, indexed by $i$ (from all ProductName in file_1_view_0).

Parameters:
- $v_i$: Value (benefit) of one unit of drug $i$ (from Value in file_1_view_0, indexed by ProductName).
- $w_i$: Weight (stock usage) of one unit of drug $i$ (from Weight in file_1_view_0, indexed by ProductName).
- $C$: Overall stock capacity (from Capacity in file_0_view_0).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of drug $i$ to order each day.

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

Data Mapping:
- $I$: All ProductName in file_1_view_0.
- $v_i$: file_1_view_0, column Value, indexed by ProductName.
- $w_i$: file_1_view_0, column Weight, indexed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.