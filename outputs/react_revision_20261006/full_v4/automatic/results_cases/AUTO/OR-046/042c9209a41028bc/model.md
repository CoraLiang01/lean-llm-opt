##### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with ProductName from products.csv)
- $x_i$ = number of units of product $i$ to order each day (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)
- $v_i$ = value (benefit) per unit of product $i$ (from Value column)
- $w_i$ = weight (stock space required) per unit of product $i$ (from Weight column)
- $C$ = overall stock capacity (from Capacity column in capacity.csv)

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

##### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)