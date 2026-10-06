ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑊: Set of warehouses (indexed by w), corresponding to all "Warehouse ID" in file_0_view_0.
- 𝑃: Set of vehicle types/products (indexed by p), corresponding to all "ProductName" in file_1_view_0.

Parameters:
- cap_w: Capacity of warehouse w. Data: file_0_view_0, column "Capacity".
- val_p: Value (benefit coefficient) of vehicle type p. Data: file_1_view_0, column "Value".
- wt_p: Storage weight/unit of vehicle type p. Data: file_1_view_0, column "Weight".

Decision Variables:
- x_{w,p}: Number of units of vehicle type p to store in warehouse w. Domain: integer, x_{w,p} ≥ 0.

Objective:
Maximize total value stored across all warehouses:
\[
\max \sum_{w \in 𝑊} \sum_{p \in 𝑃} val_p \cdot x_{w,p}
\]

Subject to:

1. Warehouse capacity constraints (for each warehouse w ∈ 𝑊):
\[
\sum_{p \in 𝑃} wt_p \cdot x_{w,p} \leq cap_w
\]

2. Nonnegativity and integrality:
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w \in 𝑊,\, p \in 𝑃
\]

DATA MAPPING

- 𝑊: All "Warehouse ID" in file_0_view_0.
- 𝑃: All "ProductName" in file_1_view_0.
- cap_w: file_0_view_0, column "Capacity", keyed by "Warehouse ID".
- val_p: file_1_view_0, column "Value", keyed by "ProductName".
- wt_p: file_1_view_0, column "Weight", keyed by "ProductName".