Abstract Mathematical Model

Sets:
- $I$: set of areas (indexed by $i$), with area identifiers from file_1_view_0.ProductName.

Parameters:
- $b_i$: benefit coefficient for area $i$ (from file_1_view_0.Value).
- $w_i$: development unit weight for area $i$ (from file_1_view_0.Weight).
- $C$: overall development capacity (from file_0_view_0.Capacity).

Decision Variables:
- $x_i$: integer, scale of development in area $i$ per day.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All file_1_view_0.ProductName rows.
- $b_i$: file_1_view_0.Value, matched to area $i$ by file_1_view_0.ProductName.
- $w_i$: file_1_view_0.Weight, matched to area $i$ by file_1_view_0.ProductName.
- $C$: file_0_view_0.Capacity (single value, all rows).
- $x_i$: Decision variable for each $i$ in $I$.

All parameters and sets are defined directly from the returned CSVQA data, preserving original file and column names.