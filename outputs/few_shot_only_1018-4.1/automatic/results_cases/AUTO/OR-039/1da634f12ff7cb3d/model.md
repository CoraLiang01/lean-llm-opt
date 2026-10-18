Let $x_{ij}$ be the number of units of vehicle type $i$ stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Indices:
- $i$ indexes vehicle types (from ProductName in products.csv)
- $j$ indexes warehouses (from Warehouse ID in capacity.csv)

Parameters:
- $p_i$: Value of vehicle type $i$ (from Value in products.csv)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from Weight in products.csv)
- $C_j$: Capacity of warehouse $j$ (from Capacity in capacity.csv)

Sets:
- Vehicle types: 
  1. Sedans
  2. SUVs
  3. Electric Vehicles
  4. Hybrid Vehicles
  5. Trucks
  6. Sports Cars
  7. Compact Cars
  8. Luxury Sedans
  9. Vans
  10. Pickup Trucks

- Warehouses:
  1. Warehouse 1
  2. Warehouse 2
  3. Warehouse 3
  4. Warehouse 4
  5. Warehouse 5
  6. Warehouse 6
  7. Warehouse 7
  8. Warehouse 8
  9. Warehouse 9
  10. Warehouse 10

Parameters (from data):

| ProductName         | $p_i$ | $w_i$ |
|---------------------|-------|-------|
| Sedans              | 1200  | 20    |
| SUVs                | 1800  | 15    |
| Electric Vehicles   | 2500  | 25    |
| Hybrid Vehicles     | 2000  | 18    |
| Trucks              | 1500  | 10    |
| Sports Cars         | 3000  | 5     |
| Compact Cars        | 1000  | 22    |
| Luxury Sedans       | 3500  | 8     |
| Vans                | 1600  | 12    |
| Pickup Trucks       | 1700  | 7     |

| Warehouse ID  | $C_j$ |
|---------------|-------|
| Warehouse 1   | 100   |
| Warehouse 2   | 80    |
| Warehouse 3   | 120   |
| Warehouse 4   | 90    |
| Warehouse 5   | 50    |
| Warehouse 6   | 30    |
| Warehouse 7   | 110   |
| Warehouse 8   | 40    |
| Warehouse 9   | 60    |
| Warehouse 10  | 35    |

Model:

Objective:
\[
\max \sum_{i} \sum_{j} p_i \cdot x_{ij}
\]

Subject to:

For each warehouse $j$:
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $x_{ij}$: number of units of vehicle type $i$ stored in warehouse $j$
- $p_i$: value of vehicle type $i$ (see table above)
- $w_i$: weight (space requirement) of vehicle type $i$ (see table above)
- $C_j$: capacity of warehouse $j$ (see table above)

All indices and coefficients are as given in the retrieved data.