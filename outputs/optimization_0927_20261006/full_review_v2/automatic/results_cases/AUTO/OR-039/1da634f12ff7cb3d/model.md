Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $i$ index vehicle types (from products.csv):

- Sedans
- SUVs
- Electric Vehicles
- Hybrid Vehicles
- Trucks
- Sports Cars
- Compact Cars
- Luxury Sedans
- Vans
- Pickup Trucks

Let $j$ index warehouses (from capacity.csv):

- Warehouse 1
- Warehouse 2
- Warehouse 3
- Warehouse 4
- Warehouse 5
- Warehouse 6
- Warehouse 7
- Warehouse 8
- Warehouse 9
- Warehouse 10

Let $v_i$ be the value per unit of vehicle type $i$:

- Sedans: $1200$
- SUVs: $1800$
- Electric Vehicles: $2500$
- Hybrid Vehicles: $2000$
- Trucks: $1500$
- Sports Cars: $3000$
- Compact Cars: $1000$
- Luxury Sedans: $3500$
- Vans: $1600$
- Pickup Trucks: $1700$

Let $w_i$ be the weight per unit of vehicle type $i$:

- Sedans: $20$
- SUVs: $15$
- Electric Vehicles: $25$
- Hybrid Vehicles: $18$
- Trucks: $10$
- Sports Cars: $5$
- Compact Cars: $22$
- Luxury Sedans: $8$
- Vans: $12$
- Pickup Trucks: $7$

Let $C_j$ be the capacity of warehouse $j$:

- Warehouse 1: $100$
- Warehouse 2: $80$
- Warehouse 3: $120$
- Warehouse 4: $90$
- Warehouse 5: $50$
- Warehouse 6: $30$
- Warehouse 7: $110$
- Warehouse 8: $40$
- Warehouse 9: $60$
- Warehouse 10: $35$

The complete mathematical model is:

Objective:
\[
\max \sum_{i \in \text{Products}} \sum_{j \in \text{Warehouses}} v_i \cdot x_{ij}
\]

Subject to, for each warehouse $j$:
\[
\sum_{i \in \text{Products}} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- Products and their parameters are as listed above.
- Warehouses and their capacities are as listed above.
- All $x_{ij}$ are nonnegative integers.