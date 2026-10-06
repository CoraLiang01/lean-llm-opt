ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of products, indexed by $i$. (Data: file_1_view_0, column ProductName)

Parameters:
- $v_i$: Value (benefit) per unit of product $i$. (Data: file_1_view_0, column Value, key ProductName)
- $w_i$: Weight (stock space required) per unit of product $i$. (Data: file_1_view_0, column Weight, key ProductName)
- $C$: Total stock capacity. (Data: file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of units of product $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity.

Each product $i$ is identified by its ProductName from products.csv. The model maximizes total benefit from ordering products, subject to the overall stock capacity from capacity.csv. All variables are nonnegative integers.