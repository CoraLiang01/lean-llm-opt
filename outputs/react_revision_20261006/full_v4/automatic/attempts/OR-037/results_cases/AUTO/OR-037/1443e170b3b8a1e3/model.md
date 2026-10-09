Mathematical Model

Sets:
- $I$: set of vehicle types (indexed by $i$), from products.csv ProductName.

Parameters:
- $p_i$: profit per unit of vehicle $i$ (products.csv Value, for $i \in I$)
- $w_i$: inventory weight per unit of vehicle $i$ (products.csv Weight, for $i \in I$)
- $C$: total inventory capacity (capacity.csv Capacity)

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$, for $i \in I$)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$