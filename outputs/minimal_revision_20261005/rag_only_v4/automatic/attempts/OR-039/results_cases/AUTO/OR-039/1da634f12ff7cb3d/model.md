ABSTRACT MATHEMATICAL MODEL

Sets:
- W: Set of warehouses (indexed by w), from file_0_view_0["Warehouse ID"]
- P: Set of vehicle types/products (indexed by p), from file_1_view_0["ProductName"]

Parameters:
- cap_w: Capacity of warehouse w  
  Data: file_0_view_0["Capacity"] for warehouse w
- val_p: Value (benefit coefficient) of vehicle type p  
  Data: file_1_view_0["Value"] for product p
- wt_p: Storage weight/unit of vehicle type p  
  Data: file_1_view_0["Weight"] for product p

Decision Variables:
- x_{w,p}: Number of units of vehicle type p to store in warehouse w  
  Domain: nonnegative integers (x_{w,p} ∈ ℤ₊)

Objective:
Maximize total value stored across all warehouses:
\[
\max \sum_{w \in W} \sum_{p \in P} val_p \cdot x_{w,p}
\]

Constraints:
1. Warehouse capacity: For each warehouse w ∈ W,
\[
\sum_{p \in P} wt_p \cdot x_{w,p} \leq cap_w
\]
2. Nonnegativity and integrality:
\[
x_{w,p} \in \mathbb{Z}_+, \quad \forall w \in W,\, p \in P
\]

DATA MAPPING

- Set W: file_0_view_0["Warehouse ID"]
- Set P: file_1_view_0["ProductName"]
- Parameter cap_w: file_0_view_0["Capacity"] (warehouse w)
- Parameter val_p: file_1_view_0["Value"] (product p)
- Parameter wt_p: file_1_view_0["Weight"] (product p)
- Decision variable x_{w,p}: Number of units of product p stored in warehouse w (integer, ≥0)