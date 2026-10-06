ABSTRACT MATHEMATICAL MODEL

Index Sets:
- W: Set of warehouses (Warehouse ID from file_0_view_0)
- P: Set of vehicle types (ProductName from file_1_view_0)

Parameters:
- cap_w: Capacity of warehouse w ∈ W (Capacity from file_0_view_0, indexed by Warehouse ID)
- val_p: Value per unit of vehicle type p ∈ P (Value from file_1_view_0, indexed by ProductName)
- wt_p: Weight (space consumed) per unit of vehicle type p ∈ P (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- x_{w,p}: Number of units of vehicle type p to store in warehouse w (integer, x_{w,p} ≥ 0)

Objective:
Maximize total value stored across all warehouses and vehicle types:
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

Data Mapping:

- W (warehouses): Warehouse ID from file_0_view_0 (capacity.csv)
- P (vehicle types): ProductName from file_1_view_0 (products.csv)
- cap_w: Capacity column from file_0_view_0, indexed by Warehouse ID
- val_p: Value column from file_1_view_0, indexed by ProductName
- wt_p: Weight column from file_1_view_0, indexed by ProductName

Each x_{w,p} is the integer number of vehicles of type p to store in warehouse w, maximizing total value while respecting each warehouse's capacity.