Let W be the set of warehouses and P be the set of vehicle types (products).

Sets:
- W = {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}
- P = {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}

Parameters:
- Value_p: Value per unit of product p (from products.csv)
  - Sedans: 1200
  - SUVs: 1800
  - Electric Vehicles: 2500
  - Hybrid Vehicles: 2000
  - Trucks: 1500
  - Sports Cars: 3000
  - Compact Cars: 1000
  - Luxury Sedans: 3500
  - Vans: 1600
  - Pickup Trucks: 1700
- Weight_p: Storage weight per unit of product p (from products.csv)
  - Sedans: 20
  - SUVs: 15
  - Electric Vehicles: 25
  - Hybrid Vehicles: 18
  - Trucks: 10
  - Sports Cars: 5
  - Compact Cars: 22
  - Luxury Sedans: 8
  - Vans: 12
  - Pickup Trucks: 7
- Capacity_w: Capacity of warehouse w (from capacity.csv)
  - Warehouse 1: 100
  - Warehouse 2: 80
  - Warehouse 3: 120
  - Warehouse 4: 90
  - Warehouse 5: 50
  - Warehouse 6: 30
  - Warehouse 7: 110
  - Warehouse 8: 40
  - Warehouse 9: 60
  - Warehouse 10: 35

Decision Variables:
- x_{w,p}: Number of units of product p to store in warehouse w per day (integer, x_{w,p} ≥ 0, integer for all w ∈ W, p ∈ P)

Objective:
Maximize total value stored across all warehouses:
\[
\text{Maximize} \quad Z = \sum_{w \in W} \sum_{p \in P} \text{Value}_p \cdot x_{w,p}
\]

Subject to:

For each warehouse w ∈ W (capacity constraints):
\[
\sum_{p \in P} \text{Weight}_p \cdot x_{w,p} \leq \text{Capacity}_w
\]
Explicitly, for each warehouse:

- Warehouse 1: 20 x_{1,Sedans} + 15 x_{1,SUVs} + 25 x_{1,Electric Vehicles} + 18 x_{1,Hybrid Vehicles} + 10 x_{1,Trucks} + 5 x_{1,Sports Cars} + 22 x_{1,Compact Cars} + 8 x_{1,Luxury Sedans} + 12 x_{1,Vans} + 7 x_{1,Pickup Trucks} ≤ 100
- Warehouse 2: (same as above, ≤ 80)
- Warehouse 3: (same as above, ≤ 120)
- Warehouse 4: (same as above, ≤ 90)
- Warehouse 5: (same as above, ≤ 50)
- Warehouse 6: (same as above, ≤ 30)
- Warehouse 7: (same as above, ≤ 110)
- Warehouse 8: (same as above, ≤ 40)
- Warehouse 9: (same as above, ≤ 60)
- Warehouse 10: (same as above, ≤ 35)

Variable domains:
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w \in W, p \in P
\]

Summary:
Maximize
\[
\sum_{w \in W} \sum_{p \in P} \text{Value}_p \cdot x_{w,p}
\]
subject to, for each warehouse w,
\[
\sum_{p \in P} \text{Weight}_p \cdot x_{w,p} \leq \text{Capacity}_w
\]
and
\[
x_{w,p} \geq 0, \text{ integer}
\]
for all warehouses w and products p.