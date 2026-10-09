Let  
$x_{ij}$ = number of units of vehicle type $i$ (ProductName from products.csv) to store in warehouse $j$ (Warehouse ID from capacity.csv), for all $i = 1,\ldots,10$, $j = 1,\ldots,10$.

Parameters:  
Let $v_i$ = Value of vehicle type $i$ (from products.csv)  
Let $w_i$ = Weight of vehicle type $i$ (from products.csv)  
Let $C_j$ = Capacity of warehouse $j$ (from capacity.csv)

**Objective:**  
Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij}
$$

**Subject to:**

For each warehouse $j$ (using Warehouse ID from capacity.csv):

Capacity constraints:
$$
\sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Where:**

From products.csv:

| $i$ | ProductName           | $v_i$ (Value) | $w_i$ (Weight) |
|-----|-----------------------|---------------|----------------|
| 1   | Sedans                | 1200          | 20             |
| 2   | SUVs                  | 1800          | 15             |
| 3   | Electric Vehicles     | 2500          | 25             |
| 4   | Hybrid Vehicles       | 2000          | 18             |
| 5   | Trucks                | 1500          | 10             |
| 6   | Sports Cars           | 3000          | 5              |
| 7   | Compact Cars          | 1000          | 22             |
| 8   | Luxury Sedans         | 3500          | 8              |
| 9   | Vans                  | 1600          | 12             |
| 10  | Pickup Trucks         | 1700          | 7              |

From capacity.csv:

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

**Decision variables:**  
$x_{ij}$: integer, $\geq 0$, for all $i = 1,\ldots,10$, $j = 1,\ldots,10$.

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij} \\
\text{s.t.}\quad
& \sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j \qquad \forall j = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,10
\end{align*}
$$

With $v_i$, $w_i$, $C_j$ as specified above, and $x_{ij}$ as the number of units of vehicle type $i$ to store in warehouse $j$.