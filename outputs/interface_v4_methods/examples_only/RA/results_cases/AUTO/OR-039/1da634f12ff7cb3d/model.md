ABSTRACT MATHEMATICAL MODEL

Index Sets:
- W: Set of warehouses (Warehouse ID from file_0_view_0)
- P: Set of vehicle types (ProductName from file_1_view_0)

Parameters:
- cap_w: Capacity of warehouse w ∈ W (Capacity from file_0_view_0, indexed by Warehouse ID)
- val_p: Value (benefit coefficient) of vehicle type p ∈ P (Value from file_1_view_0, indexed by ProductName)
- wt_p: Weight (space consumption per unit) of vehicle type p ∈ P (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- x_{w,p}: Number of vehicles of type p to store in warehouse w (integer, x_{w,p} ≥ 0)

Objective:
Maximize total value:
\[
\max \sum_{w \in W} \sum_{p \in P} val_p \cdot x_{w,p}
\]

Constraints:
1. Warehouse capacity constraints (for each warehouse w ∈ W):
\[
\sum_{p \in P} wt_p \cdot x_{w,p} \leq cap_w
\]

2. Nonnegativity and integrality:
\[
x_{w,p} \in \mathbb{Z}_+, \quad \forall w \in W,\, p \in P
\]

DATA MAPPING

- W (warehouses): Warehouse ID from file_0_view_0 (capacity.csv)
- P (vehicle types): ProductName from file_1_view_0 (products.csv)
- cap_w: Capacity from file_0_view_0, indexed by Warehouse ID
- val_p: Value from file_1_view_0, indexed by ProductName
- wt_p: Weight from file_1_view_0, indexed by ProductName

Each x_{w,p} is the integer number of vehicles of type p to store in warehouse w. All parameters are mapped directly from the supplied files as described above.