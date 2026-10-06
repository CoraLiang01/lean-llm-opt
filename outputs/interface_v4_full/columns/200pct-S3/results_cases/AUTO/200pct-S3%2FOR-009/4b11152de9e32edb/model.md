ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas available for development (from products.csv, column ProductName).

Parameters:
- $b_i$: Development benefit per unit in area $i$ (from products.csv, column Value, table_id: file_1_view_0).
- $w_i$: Resource requirement per unit in area $i$ (from products.csv, column Weight, table_id: file_1_view_0).
- $C$: Overall development capacity (from capacity.csv, column Capacity, table_id: file_0_view_0).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$.

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping:
- $I$: file_1_view_0.ProductName
- $b_i$: file_1_view_0.Value
- $w_i$: file_1_view_0.Weight
- $C$: file_0_view_0.Capacity

All parameters and index sets are mapped directly from the retrieved CSV data, preserving original identifiers and columns.