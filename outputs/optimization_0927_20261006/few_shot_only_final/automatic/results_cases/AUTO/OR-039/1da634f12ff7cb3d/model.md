Let $x_{ij}$ be the number of units of vehicle type $i$ (ProductName from products.csv) to be stored in warehouse $j$ (Warehouse ID from capacity.csv). All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Parameters:**

- $V_i$: Value of vehicle type $i$ (from products.csv, column "Value")
- $w_i$: Weight (space requirement) of vehicle type $i$ (from products.csv, column "Weight")
- $C_j$: Capacity of warehouse $j$ (from capacity.csv, column "Capacity")

**Sets:**

- $i \in$ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}
- $j \in$ {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}

**Model:**

Objective:
\[
\max \sum_{i} \sum_{j} V_i \cdot x_{ij}
\]

Subject to, for each warehouse $j$:

\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \qquad \forall j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Numerical Data:**

Vehicle types and parameters (from products.csv):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Sedans                | 1200  | 20     |
| SUVs                  | 1800  | 15     |
| Electric Vehicles     | 2500  | 25     |
| Hybrid Vehicles       | 2000  | 18     |
| Trucks                | 1500  | 10     |
| Sports Cars           | 3000  | 5      |
| Compact Cars          | 1000  | 22     |
| Luxury Sedans         | 3500  | 8      |
| Vans                  | 1600  | 12     |
| Pickup Trucks         | 1700  | 7      |

Warehouses and capacities (from capacity.csv):

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

**Complete Model:**

\[
\max \sum_{i \in \text{ProductName}} \sum_{j \in \text{Warehouse ID}} V_i \cdot x_{ij}
\]

Subject to, for each $j \in$ Warehouse ID:

\[
\sum_{i \in \text{ProductName}} w_i \cdot x_{ij} \leq C_j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- $V_i$ and $w_i$ are as in the table above for each ProductName $i$,
- $C_j$ is as in the table above for each Warehouse ID $j$.