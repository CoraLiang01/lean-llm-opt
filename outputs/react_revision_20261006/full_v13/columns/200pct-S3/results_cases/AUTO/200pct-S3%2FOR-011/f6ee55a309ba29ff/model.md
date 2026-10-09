#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all ProductName in file_1_view_0)
- $x_i$ = number of units of product $i$ to order each day (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)
- $v_i$ = Value of product $i$ (from Value in file_1_view_0)
- $w_i$ = Weight of product $i$ (from Weight in file_1_view_0)
- $C$ = overall stock capacity (from Capacity in file_0_view_0)

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
- $x_i$: decision variable for each $i \in I$ (nonnegative integer)