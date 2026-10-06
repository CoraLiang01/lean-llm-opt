ABSTRACT MATHEMATICAL MODEL

Index Sets:
- I: Set of bread types (ProductName from file_1_view_0)

Parameters:
- v_i: Expected profit per unit of bread type i ∈ I (Value from file_1_view_0, column "Value")
- w_i: Storage requirement per unit of bread type i ∈ I (Weight from file_1_view_0, column "Weight")
- C: Total available storage capacity (Capacity from file_0_view_0, column "Capacity")

Decision Variables:
- x_i: Number of units of bread type i ∈ I to order each day (integer, x_i ≥ 0)

Objective:
Maximize total expected profit:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
- Storage capacity constraint:
\[
\sum_{i \in I} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

DATA MAPPING

- Index set I: file_1_view_0.ProductName
- v_i: file_1_view_0.Value (for each ProductName)
- w_i: file_1_view_0.Weight (for each ProductName)
- C: file_0_view_0.Capacity

Each x_i is indexed by file_1_view_0.ProductName. All parameters are mapped directly from the specified columns. The model maximizes total expected profit from bread orders, subject to the bakery's storage capacity.