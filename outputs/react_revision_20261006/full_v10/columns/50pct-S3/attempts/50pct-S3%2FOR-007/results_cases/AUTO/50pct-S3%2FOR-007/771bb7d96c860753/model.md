Mathematical Model

Index Sets:
- Let I be the set of vehicle types, with each i ∈ I corresponding to ProductName from file_1_view_0.

Parameters:
- v_i: profit per unit of vehicle type i (Value from file_1_view_0, column Value, indexed by ProductName)
- w_i: inventory space required per unit of vehicle type i (Weight from file_1_view_0, column Weight, indexed by ProductName)
- C: total inventory capacity (Capacity from file_0_view_0, scalar)

Decision Variables:
- x_i: number of vehicles of type i to order per day (integer, x_i ≥ 0, ∀ i ∈ I)

Objective:
Maximize total profit:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:

Inventory capacity constraint:
$$
\sum_{i \in I} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

Data Mapping

Index Sets:
- I: file_1_view_0, column ProductName

Parameters:
- v_i: file_1_view_0, column Value, indexed by ProductName
- w_i: file_1_view_0, column Weight, indexed by ProductName
- C: file_0_view_0, column Capacity (scalar)

Decision Variables:
- x_i: number of vehicles of type i to order per day (indexed by ProductName from file_1_view_0)

Objective:
- Maximize total profit: sum over i ∈ I of v_i x_i

Constraints:
- Inventory capacity: sum over i ∈ I of w_i x_i ≤ C
- x_i ∈ ℤ₊, ∀ i ∈ I