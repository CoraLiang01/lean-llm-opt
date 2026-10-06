ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas (from file_1_view_0, column ProductName)

Parameters:
- $b_i$: Benefit coefficient for area $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: Development unit weight for area $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: Overall development capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Integer, scale of development in area $i$ per day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: file_1_view_0, column ProductName
- $b_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity

All parameters and index sets are mapped directly from the retrieved CSV data, preserving original file and column names.