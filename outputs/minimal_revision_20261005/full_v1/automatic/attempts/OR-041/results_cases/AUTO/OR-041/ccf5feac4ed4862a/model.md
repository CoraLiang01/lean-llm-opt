ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of areas, indexed by $i$ (from ProductName in products.csv; e.g., Queens, Brooklyn)

Parameters:
- $v_i$: Development benefit per unit in area $i$ (Value, file_1_view_0)
- $w_i$: Resource requirement per unit in area $i$ (Weight, file_1_view_0)
- $C$: Overall development capacity (Capacity, file_0_view_0)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

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
- $v_i$: Value column in file_1_view_0 (products.csv), for each $i$
- $w_i$: Weight column in file_1_view_0 (products.csv), for each $i$
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i$ in $I$ (nonnegative integer)

All parameters and sets are mapped directly from the returned CSVQA data, preserving original identifiers and file/column names.