ABSTRACT MATHEMATICAL MODEL

Index Sets:
- I: Set of vehicle types (ProductName from products.csv)

Parameters:
- v_i: Profit per unit of vehicle type i ∈ I (Value column, products.csv)
- w_i: Inventory weight per unit of vehicle type i ∈ I (Weight column, products.csv)
- C: Overall inventory capacity (Capacity column, capacity.csv)

Decision Variables:
- x_i: Number of vehicles of type i ∈ I to order per day (nonnegative integer)

Objective:
Maximize total profit:
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
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

DATA MAPPING

- I: All rows in file_1_view_0 (products.csv), ProductName
- v_i: file_1_view_0 (products.csv), Value column, keyed by ProductName
- w_i: file_1_view_0 (products.csv), Weight column, keyed by ProductName
- C: file_0_view_0 (capacity.csv), Capacity column (single value)
- x_i: Decision variable for each ProductName in file_1_view_0

All data is used as returned, with no omitted or synthesized values.