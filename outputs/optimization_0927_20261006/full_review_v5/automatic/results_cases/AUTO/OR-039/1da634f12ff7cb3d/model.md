Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let:
- $I$ = set of vehicle types (indexed by ProductName)
- $J$ = set of warehouses (indexed by Warehouse ID)
- $p_i$ = Value of vehicle type $i$
- $w_i$ = Weight (space requirement) of vehicle type $i$
- $C_j$ = Capacity of warehouse $j$

#### Sets and Parameters

Vehicle Types ($i$):
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

Warehouses ($j$):
| Warehouse ID   | Capacity |
|---------------|----------|
| Warehouse 1   | 100      |
| Warehouse 2   | 80       |
| Warehouse 3   | 120      |
| Warehouse 4   | 90       |
| Warehouse 5   | 50       |
| Warehouse 6   | 30       |
| Warehouse 7   | 110      |
| Warehouse 8   | 40       |
| Warehouse 9   | 60       |
| Warehouse 10  | 35       |

#### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ stored in warehouse $j$.

#### Objective Function

$$
\max \sum_{i \in I} \sum_{j \in J} p_i \cdot x_{ij}
$$

#### Constraints

For each warehouse $j \in J$:
$$
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Explicitly, with identifiers:

Let $I$ = {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}

Let $J$ = {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}

For each $j$:

- Warehouse 1: $\sum_{i \in I} w_i x_{i, \text{Warehouse 1}} \leq 100$
- Warehouse 2: $\sum_{i \in I} w_i x_{i, \text{Warehouse 2}} \leq 80$
- Warehouse 3: $\sum_{i \in I} w_i x_{i, \text{Warehouse 3}} \leq 120$
- Warehouse 4: $\sum_{i \in I} w_i x_{i, \text{Warehouse 4}} \leq 90$
- Warehouse 5: $\sum_{i \in I} w_i x_{i, \text{Warehouse 5}} \leq 50$
- Warehouse 6: $\sum_{i \in I} w_i x_{i, \text{Warehouse 6}} \leq 30$
- Warehouse 7: $\sum_{i \in I} w_i x_{i, \text{Warehouse 7}} \leq 110$
- Warehouse 8: $\sum_{i \in I} w_i x_{i, \text{Warehouse 8}} \leq 40$
- Warehouse 9: $\sum_{i \in I} w_i x_{i, \text{Warehouse 9}} \leq 60$
- Warehouse 10: $\sum_{i \in I} w_i x_{i, \text{Warehouse 10}} \leq 35$

Where $w_i$ is as in the table above.

All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

#### Summary Table of Parameters

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

| Warehouse ID   | Capacity |
|---------------|----------|
| Warehouse 1   | 100      |
| Warehouse 2   | 80       |
| Warehouse 3   | 120      |
| Warehouse 4   | 90       |
| Warehouse 5   | 50       |
| Warehouse 6   | 30       |
| Warehouse 7   | 110      |
| Warehouse 8   | 40       |
| Warehouse 9   | 60       |
| Warehouse 10  | 35       |