ABSTRACT MATHEMATICAL MODEL

Index Sets:
- I: Set of produce types (ProductName from products.csv, table_id: file_1_view_0)

Parameters:
- v_i: Benefit per unit of produce i (Value column, file_1_view_0)
- w_i: Weight per unit of produce i (Weight column, file_1_view_0)
- C: Overall inventory capacity (Capacity column, file_0_view_0)

Decision Variables:
- x_i: Number of units of produce i to order daily (integer, x_i ≥ 0, ∀i ∈ I)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in I} v_i x_i
\]

Constraints:
1. Inventory capacity constraint:
\[
\sum_{i \in I} w_i x_i \leq C
\]
2. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

DATA MAPPING

- I: All rows in file_1_view_0 (products.csv), indexed by ProductName.
- v_i: file_1_view_0, column Value, for each ProductName.
- w_i: file_1_view_0, column Weight, for each ProductName.
- C: file_0_view_0, column Capacity (single value).

Each x_i is indexed by ProductName from products.csv (file_1_view_0). All parameters are mapped directly from the specified columns and table_ids.