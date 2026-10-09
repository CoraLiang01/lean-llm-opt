ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of products (indexed by $i$), from ProductName in products.csv.

Parameters:
- $w_i$: weight (stock space requirement) of product $i$, from Weight in products.csv.
- $v_i$: value (benefit) of product $i$, from Value in products.csv.
- $C$: total stock capacity, from Capacity in capacity.csv.

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
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

DATA MAPPING

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value
- $x_i$: Decision variable for each $i \in I$