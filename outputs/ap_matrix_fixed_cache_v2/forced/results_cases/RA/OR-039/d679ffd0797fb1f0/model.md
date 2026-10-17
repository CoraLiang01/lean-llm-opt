Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $i$ index vehicle types (from products.csv), and $j$ index warehouses (from capacity.csv).

Define:
- $v_i$ = Value of vehicle type $i$
- $w_i$ = Weight (space requirement) of vehicle type $i$
- $C_j$ = Capacity of warehouse $j$

**Vehicle Types and Parameters (from products.csv):**

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

**Warehouses and Capacities (from capacity.csv):**

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

---

**Mathematical Model**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij}
$$

**Warehouse Capacity Constraints:**
For each warehouse $j$:
$$
\sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j \quad \forall j \in \{1,\ldots,10\}
$$

**Non-negativity and Integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

---

**Parameter Values (explicit):**

- $v_1 = 1200$, $w_1 = 20$  (Sedans)
- $v_2 = 1800$, $w_2 = 15$  (SUVs)
- $v_3 = 2500$, $w_3 = 25$  (Electric Vehicles)
- $v_4 = 2000$, $w_4 = 18$  (Hybrid Vehicles)
- $v_5 = 1500$, $w_5 = 10$  (Trucks)
- $v_6 = 3000$, $w_6 = 5$  (Sports Cars)
- $v_7 = 1000$, $w_7 = 22$  (Compact Cars)
- $v_8 = 3500$, $w_8 = 8$  (Luxury Sedans)
- $v_9 = 1600$, $w_9 = 12$  (Vans)
- $v_{10} = 1700$, $w_{10} = 7$ (Pickup Trucks)

- $C_1 = 100$ (Warehouse 1)
- $C_2 = 80$ (Warehouse 2)
- $C_3 = 120$ (Warehouse 3)
- $C_4 = 90$ (Warehouse 4)
- $C_5 = 50$ (Warehouse 5)
- $C_6 = 30$ (Warehouse 6)
- $C_7 = 110$ (Warehouse 7)
- $C_8 = 40$ (Warehouse 8)
- $C_9 = 60$ (Warehouse 9)
- $C_{10} = 35$ (Warehouse 10)

---

**Complete Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij} \\
\text{s.t.}\quad & \sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j \quad \forall j = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,10
\end{align*}
$$

Where all $v_i$, $w_i$, and $C_j$ are as listed above, and $x_{ij}$ is the number of vehicles of type $i$ to store in warehouse $j$.