Mathematical Model

Sets:
- $I$: set of products, indexed by $i$ (from ProductName in file_1_view_0)

Parameters:
- $v_i$: value (benefit) per unit of product $i$ (Value from file_1_view_0, column Value)
- $w_i$: weight (stock space required) per unit of product $i$ (Weight from file_1_view_0, column Weight)
- $C$: total stock capacity (Capacity from file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ to order each day

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

- $I$: All records in file_1_view_0 (ProductName)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$ (ProductName)