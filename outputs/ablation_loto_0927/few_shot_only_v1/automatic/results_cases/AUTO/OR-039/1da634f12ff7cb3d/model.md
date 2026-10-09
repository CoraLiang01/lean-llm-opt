**Mathematical Model for New Car Sales in Norway Inventory-Replenishment**

Let:
- $i$ index the vehicle types (from the "products.csv" file, using ProductName).
- $j$ index the warehouses (from the "capacity.csv" file, using Warehouse ID).
- $v_i$ = Value of vehicle type $i$.
- $w_i$ = Weight (space requirement) of vehicle type $i$.
- $C_j$ = Capacity of warehouse $j$.
- $x_{ij}$ = Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable).

**Objective:**
\[
\max \sum_{i} \sum_{j} v_i \cdot x_{ij}
\]

**Subject to:**

For each warehouse $j$ (using Warehouse ID as in the data):

\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \qquad \forall j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameters from the data:**

- **Warehouses and Capacities:**

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

- **Vehicle Types, Values, and Weights:**

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

---

**Full Formulation:**

**Decision variables:**
\[
x_{ij} = \text{number of units of vehicle type } i \text{ to store in warehouse } j, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Objective:**
\[
\max \sum_{i \in \{\text{Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}\}} \sum_{j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}} v_i \cdot x_{ij}
\]

**Warehouse capacity constraints (for each $j$):**
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j
\]

where:
- $v_i$ and $w_i$ are as in the table above,
- $C_j$ is the capacity for warehouse $j$ as in the table above.

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]