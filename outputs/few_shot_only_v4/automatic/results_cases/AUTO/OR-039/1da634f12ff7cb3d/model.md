Let $x_{ij}$ be the number of vehicles of type $i$ stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let:
- $i$ index the vehicle types (from ProductName in products.csv)
- $j$ index the warehouses (from Warehouse ID in capacity.csv)
- $p_i$ = Value of vehicle type $i$
- $w_i$ = Weight of vehicle type $i$ (space required per unit)
- $C_j$ = Capacity of warehouse $j$

**Sets:**
- Vehicle types $i \in \{$Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks$\}$
- Warehouses $j \in \{$Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10$\}$

**Parameters:**

| ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Sedans              | 1200          | 20             |
| SUVs                | 1800          | 15             |
| Electric Vehicles   | 2500          | 25             |
| Hybrid Vehicles     | 2000          | 18             |
| Trucks              | 1500          | 10             |
| Sports Cars         | 3000          | 5              |
| Compact Cars        | 1000          | 22             |
| Luxury Sedans       | 3500          | 8              |
| Vans                | 1600          | 12             |
| Pickup Trucks       | 1700          | 7              |

| Warehouse ID  | $C_j$ (Capacity) |
|---------------|:----------------:|
| Warehouse 1   | 100              |
| Warehouse 2   | 80               |
| Warehouse 3   | 120              |
| Warehouse 4   | 90               |
| Warehouse 5   | 50               |
| Warehouse 6   | 30               |
| Warehouse 7   | 110              |
| Warehouse 8   | 40               |
| Warehouse 9   | 60               |
| Warehouse 10  | 35               |

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all vehicle types $i$ and warehouses $j$

**Objective:**
\[
\max \sum_{i} \sum_{j} p_i \cdot x_{ij}
\]

**Subject to:**

For each warehouse $j$:
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}
\]

For all vehicle types $i$ and warehouses $j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Where:**

- $p_i$, $w_i$ as above for each vehicle type $i$
- $C_j$ as above for each warehouse $j$
- $x_{ij}$ is the number of vehicles of type $i$ stored in warehouse $j$