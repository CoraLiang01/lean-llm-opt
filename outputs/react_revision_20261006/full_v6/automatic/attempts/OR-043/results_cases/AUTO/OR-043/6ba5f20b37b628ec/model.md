Mathematical Optimization Model

Index Sets:
- $I$: Set of drug products, indexed by $i$, with identifiers ProductName from file_1_view_0.

Parameters:
- $v_i$: Value (benefit) of ordering one unit of drug $i$ (from file_1_view_0, column Value).
- $w_i$: Weight (stock space required) for one unit of drug $i$ (from file_1_view_0, column Weight).
- $C$: Overall stock capacity (from file_0_view_0, column Capacity).

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
- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (ProductName)