#### Mathematical Model

Let $I$ be the set of products with identifiers ProductName from file_1_view_0.

Parameters:
- $v_i$: Value of product $i$ (from column Value in file_1_view_0)
- $w_i$: Weight of product $i$ (from column Weight in file_1_view_0)
- $C$: Overall stock capacity (from column Capacity in file_0_view_0)

Decision variables:
- $x_i$: Number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

#### Data Mapping

- $I$: All ProductName in file_1_view_0
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (ProductName in file_1_view_0)