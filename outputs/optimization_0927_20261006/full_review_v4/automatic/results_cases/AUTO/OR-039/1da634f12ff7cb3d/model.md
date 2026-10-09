Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $I$ be the set of vehicle types (indexed by ProductName), and $J$ be the set of warehouses (indexed by Warehouse ID).

Let $v_i$ be the Value of vehicle type $i$, and $w_i$ be the Weight of vehicle type $i$ (space consumed per unit).
Let $C_j$ be the Capacity of warehouse $j$.

---

**Sets:**

- $I =$ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}
- $J =$ {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}

**Parameters:**

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

| Warehouse ID  | Capacity |
|--------------|----------|
| Warehouse 1  | 100      |
| Warehouse 2  | 80       |
| Warehouse 3  | 120      |
| Warehouse 4  | 90       |
| Warehouse 5  | 50       |
| Warehouse 6  | 30       |
| Warehouse 7  | 110      |
| Warehouse 8  | 40       |
| Warehouse 9  | 60       |
| Warehouse 10 | 35       |

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} = \text{number of units of vehicle type } i \text{ to store in warehouse } j, \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, \forall j \in J
$$

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_i \cdot x_{ij}
$$

**Subject to:**

For each warehouse $j \in J$ (using the original Warehouse ID):

$$
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j, \quad \forall j \in J
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, \forall j \in J
$$

---

**Where:**

- $v_i$ and $w_i$ are as given in the table above for each ProductName.
- $C_j$ is as given in the table above for each Warehouse ID.

All data and identifiers are preserved in source order.