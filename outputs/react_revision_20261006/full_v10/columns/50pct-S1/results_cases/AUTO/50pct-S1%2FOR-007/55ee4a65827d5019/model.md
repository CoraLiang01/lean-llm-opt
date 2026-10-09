Mathematical Model

Index Sets:
- Let I be the set of vehicle types (ProductName from file_1_view_0).

Parameters:
- For each i in I:
    - v_i: profit per unit of vehicle type i (Value from file_1_view_0)
    - w_i: inventory weight per unit of vehicle type i (Weight from file_1_view_0)
- C: overall inventory capacity (Capacity from file_0_view_0)

Decision Variables:
- For each i in I:
    - x_i: number of vehicles of type i to order per day (x_i ∈ ℤ≥0)

Objective:
Maximize total profit:
$$
\max \sum_{i \in I} v_i x_i
$$

Constraints:
Overall inventory capacity:
$$
\sum_{i \in I} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- I: All ProductName values in file_1_view_0 (products.csv)
- v_i: Value column in file_1_view_0, keyed by ProductName
- w_i: Weight column in file_1_view_0, keyed by ProductName
- C: Capacity in file_0_view_0 (capacity.csv), row 0
- x_i: Decision variable for each i ∈ I

All parameters and index sets are mapped directly from the current CSVQA data.