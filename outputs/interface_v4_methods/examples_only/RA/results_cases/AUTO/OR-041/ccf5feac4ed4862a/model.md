ABSTRACT MATHEMATICAL MODEL

Index Sets:
- I: Set of areas available for development (from products.csv, column ProductName)

Parameters:
- b_i: Development benefit per unit scale in area i ∈ I (from products.csv, column Value, table_id: file_1_view_0)
- w_i: Development resource consumption per unit scale in area i ∈ I (from products.csv, column Weight, table_id: file_1_view_0)
- C: Overall development capacity (from capacity.csv, column Capacity, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units to develop per day in area i ∈ I (x_i ∈ ℤ₊, i.e., nonnegative integers)

Objective:
Maximize total development benefit:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
- Overall development capacity constraint:
\[
\sum_{i \in I} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

DATA MAPPING

- I: All rows in file_1_view_0 (products.csv), column ProductName
- b_i: file_1_view_0 (products.csv), column Value, keyed by ProductName
- w_i: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- C: file_0_view_0 (capacity.csv), column Capacity

Each x_i is indexed by the original ProductName (area) from products.csv. All parameters are mapped directly from the supplied files and columns.