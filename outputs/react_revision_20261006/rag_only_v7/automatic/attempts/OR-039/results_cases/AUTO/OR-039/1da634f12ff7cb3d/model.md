Mathematical Model (Abstract Formulation):

Index Sets:
- W: Set of warehouses (indexed by w), from file_0_view_0["Warehouse ID"]
- P: Set of vehicle types/products (indexed by p), from file_1_view_0["ProductName"]

Parameters:
- cap_w: Capacity of warehouse w (from file_0_view_0["Capacity"])
- val_p: Value (benefit coefficient) of vehicle type p (from file_1_view_0["Value"])
- wt_p: Storage weight/unit of vehicle type p (from file_1_view_0["Weight"])

Decision Variables:
- x_p ≥ 0, integer: Number of vehicles of type p to be ordered per day

Objective:
Maximize total value of vehicles ordered:
\[
\max \sum_{p \in P} val_p \cdot x_p
\]

Subject to:
For each warehouse w ∈ W:
\[
\sum_{p \in P} wt_p \cdot x_p \leq cap_w
\]
and
\[
x_p \in \mathbb{Z}_+, \quad \forall p \in P
\]

Data Mapping:
- W = file_0_view_0["Warehouse ID"]
- P = file_1_view_0["ProductName"]
- cap_w = file_0_view_0["Capacity"], keyed by "Warehouse ID"
- val_p = file_1_view_0["Value"], keyed by "ProductName"
- wt_p = file_1_view_0["Weight"], keyed by "ProductName"
- x_p: integer, nonnegative, for each p ∈ P

Notes:
- Each warehouse imposes a separate capacity constraint on the global order vector x_p.
- All variables x_p are global (not warehouse-specific), as per the user description.