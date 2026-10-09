Mathematical Model

Index Sets:
- Let I be the set of vehicle types, with each i ∈ I corresponding to ProductName from file_1_view_0.

Parameters:
- v_i: Value (profit) of vehicle type i, from file_1_view_0, column Value.
- w_i: Weight (inventory space required) of vehicle type i, from file_1_view_0, column Weight.
- C: Overall inventory capacity, from file_0_view_0, column Capacity.

Decision Variables:
- x_i: Number of vehicles of type i to order per day. Domain: x_i ∈ ℤ≥0, ∀ i ∈ I.

Objective:
Maximize total profit:
$$
\max \sum_{i \in I} v_i x_i
$$

Constraints:
Inventory capacity constraint:
$$
\sum_{i \in I} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

Data Mapping

- I: All ProductName in file_1_view_0.
- v_i: file_1_view_0, column Value, keyed by ProductName.
- w_i: file_1_view_0, column Weight, keyed by ProductName.
- C: file_0_view_0, column Capacity.
- x_i: Decision variable for each ProductName in file_1_view_0.