ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, with business identifier ProductName from file_1_view_0 (products.csv).

Parameters:
- $v_i$: Profit per unit of vehicle type $i$, from Value in file_1_view_0 (products.csv).
- $w_i$: Inventory weight (space requirement) per unit of vehicle type $i$, from Weight in file_1_view_0 (products.csv).
- $C$: Overall inventory capacity, from Capacity in file_0_view_0 (capacity.csv).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

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

- $I$ (vehicle types): file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value
- $w_i$: file_1_view_0.Weight
- $C$: file_0_view_0.Capacity

All parameters and identifiers are used exactly as returned by the source files.