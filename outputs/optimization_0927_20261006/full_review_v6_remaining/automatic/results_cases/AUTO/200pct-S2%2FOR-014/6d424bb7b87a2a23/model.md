Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) to be placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, with ShelfID and Capacity as below (in source order):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

- Let $P$ be the set of products, with ProductName, Value, and Weight as below (in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**

Maximize the total value of products allocated to all shelves:
$$
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
$$

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $i \in S$, the total weight of products placed on shelf $i$ cannot exceed its capacity:
   $$
   \sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in S
   $$

2. **Integrality and Nonnegativity:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \forall j \in P
   $$

**Explicit Data Used:**

- Shelves (ShelfID, Capacity):  
  1 (5.0), 2 (7.0), 3 (6.0), 4 (8.0), 5 (5.5), 6 (9.0), 7 (6.5), 8 (7.5), 9 (8.2), 10 (5.7)

- Products (ProductName, Value, Weight):  
  Smartphone (200, 1.0), Laptop (1500, 5.0), Headphones (100, 0.5), Camera (800, 2.0), Smartwatch (250, 0.3), Tablet (600, 1.5), Bluetooth Speaker (150, 1.0), Keyboard (80, 0.8), Mouse (50, 0.2), Monitor (300, 3.0), Printer (400, 4.0), External Hard Drive (120, 0.5), Router (60, 0.3), Power Bank (40, 0.4), Memory Card (30, 0.05), USB Flash Drive (25, 0.02), Smart Home Hub (100, 0.6), Gaming Console (500, 4.0), Fitness Tracker (90, 0.2), E-Reader (180, 0.5)

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \forall j \in P
\end{align*}
$$

Where all sets, parameters, and coefficients are as listed above, preserving the original file and row order.