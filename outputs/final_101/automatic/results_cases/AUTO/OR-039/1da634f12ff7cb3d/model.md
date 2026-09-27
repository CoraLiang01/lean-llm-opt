Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Indices:
- $i$ indexes vehicle types (see ProductName below)
- $j$ indexes warehouses (see Warehouse ID below)

Parameters:
- $v_i$: Value of vehicle type $i$ (from Value column)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from Weight column)
- $C_j$: Capacity of warehouse $j$ (from Capacity column)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ stored in warehouse $j$

Objective:
\[
\max \sum_{i} \sum_{j} v_i \cdot x_{ij}
\]

Subject to:

For each warehouse $j$:
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Retrieved Data

Warehouses (in source order):

| Warehouse ID   | Capacity |
|----------------|----------|
| Warehouse 1    | 100      |
| Warehouse 2    | 80       |
| Warehouse 3    | 120      |
| Warehouse 4    | 90       |
| Warehouse 5    | 50       |
| Warehouse 6    | 30       |
| Warehouse 7    | 110      |
| Warehouse 8    | 40       |
| Warehouse 9    | 60       |
| Warehouse 10   | 35       |

Vehicle Types (in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedans              | 1200  | 20     |
| SUVs                | 1800  | 15     |
| Electric Vehicles   | 2500  | 25     |
| Hybrid Vehicles     | 2000  | 18     |
| Trucks              | 1500  | 10     |
| Sports Cars         | 3000  | 5      |
| Compact Cars        | 1000  | 22     |
| Luxury Sedans       | 3500  | 8      |
| Vans                | 1600  | 12     |
| Pickup Trucks       | 1700  | 7      |

All variables $x_{ij}$ are nonnegative integers. The objective is to maximize total value stored across all warehouses, subject to each warehouse's capacity.