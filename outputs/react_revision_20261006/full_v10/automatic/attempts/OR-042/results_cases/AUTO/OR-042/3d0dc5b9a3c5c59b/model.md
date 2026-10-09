Mathematical Model

Index Sets:
Let $I$ be the set of drug types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
For each $i \in I$:
- $v_i$: Value of drug type $i$ (file_1_view_0, column Value)
- $w_i$: Weight per unit of drug type $i$ (file_1_view_0, column Weight)

Let $C$ be the overall inventory capacity (file_0_view_0, column Capacity).

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of drug type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, for each ProductName
- $w_i$: file_1_view_0, column Weight, for each ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv)
- $x_i$: integer variable for each $i \in I$ (decision: number of units to order of each drug type)