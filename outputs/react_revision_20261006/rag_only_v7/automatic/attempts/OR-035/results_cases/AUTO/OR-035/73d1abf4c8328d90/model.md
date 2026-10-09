Mathematical Model (Abstract Formulation):

Index Sets:
- 𝑃: Set of bread products, indexed by i (ProductName from file_1_view_0)

Parameters:
- v_i: Expected profit per unit of product i (Value, from file_1_view_0)
- w_i: Storage weight per unit of product i (Weight, from file_1_view_0)
- C: Total available storage capacity (Capacity, from file_0_view_0)

Decision Variables:
- x_i: Number of units of bread product i to order each day (integer, x_i ≥ 0)

Objective:
Maximize total expected profit:
\[
\max \sum_{i \in P} v_i \cdot x_i
\]

Subject to:
- Storage capacity constraint:
\[
\sum_{i \in P} w_i \cdot x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in P
\]

Data Mapping:
- Index set 𝑃: All ProductName in file_1_view_0
- Parameter v_i: Value column in file_1_view_0, keyed by ProductName
- Parameter w_i: Weight column in file_1_view_0, keyed by ProductName
- Parameter C: Capacity column in file_0_view_0 (single value)
- Decision variable x_i: Number of units to order for each ProductName in file_1_view_0

This model selects integer order quantities for each bread type to maximize profit, subject to the total storage capacity.