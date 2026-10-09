Mathematical Model

Sets:
- $I$: set of produce types, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: value (benefit) per unit of produce $i$ (file_1_view_0, column Value)
- $w_i$: weight per unit of produce $i$ (file_1_view_0, column Weight)
- $C$: total inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: number of units of produce $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C$: file_0_view_0 (capacity.csv), column Capacity
- $x_i$: integer variable for each $i \in I$ (ProductName from file_1_view_0)