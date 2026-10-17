Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let:
- $i$ index vehicle types (from the "products.csv" file, in source order):
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
- $j$ index warehouses (from the "capacity.csv" file, in source order):
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

Let $v_i$ be the value of vehicle type $i$ (from "Value" column).
Let $w_i$ be the weight (space requirement) of vehicle type $i$ (from "Weight" column).
Let $C_j$ be the capacity of warehouse $j$ (from "Capacity" column).

The complete mathematical model is:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij}
\]

Subject to, for each warehouse $j$ (for $j=1,\ldots,10$):
\[
\sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,10
\]

Where:

Vehicle types and parameters (from "products.csv", in source order):

| $i$ | ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|---------------|
| 1   | Sedans              | 1200          | 20            |
| 2   | SUVs                | 1800          | 15            |
| 3   | Electric Vehicles   | 2500          | 25            |
| 4   | Hybrid Vehicles     | 2000          | 18            |
| 5   | Trucks              | 1500          | 10            |
| 6   | Sports Cars         | 3000          | 5             |
| 7   | Compact Cars        | 1000          | 22            |
| 8   | Luxury Sedans       | 3500          | 8             |
| 9   | Vans                | 1600          | 12            |
| 10  | Pickup Trucks       | 1700          | 7             |

Warehouses and capacities (from "capacity.csv", in source order):

| $j$ | Warehouse ID   | $C_j$ (Capacity) |
|-----|---------------|------------------|
| 1   | Warehouse 1   | 100              |
| 2   | Warehouse 2   | 80               |
| 3   | Warehouse 3   | 120              |
| 4   | Warehouse 4   | 90               |
| 5   | Warehouse 5   | 50               |
| 6   | Warehouse 6   | 30               |
| 7   | Warehouse 7   | 110              |
| 8   | Warehouse 8   | 40               |
| 9   | Warehouse 9   | 60               |
| 10  | Warehouse 10  | 35               |

Decision variables:
\[
x_{ij} = \text{number of units of vehicle type $i$ to be stored in warehouse $j$},\quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Summary of the model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{10} v_i x_{ij} \\
\text{s.t.}\ & \sum_{i=1}^{10} w_i x_{ij} \leq C_j, \quad \forall j=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0},\quad \forall i=1,\ldots,10;\ j=1,\ldots,10
\end{align*}
\]