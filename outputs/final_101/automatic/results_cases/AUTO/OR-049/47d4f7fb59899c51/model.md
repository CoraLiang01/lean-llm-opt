Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Shelves ($i$):  
  1 (Capacity 5.0),  
  2 (Capacity 7.0),  
  3 (Capacity 6.0),  
  4 (Capacity 8.0),  
  5 (Capacity 5.5),  
  6 (Capacity 9.0),  
  7 (Capacity 6.5),  
  8 (Capacity 7.5),  
  9 (Capacity 8.2),  
  10 (Capacity 5.7)

- Products ($j$):  
  Smartphone (Value 200, Weight 1.0)  
  Laptop (Value 1500, Weight 5.0)  
  Headphones (Value 100, Weight 0.5)  
  Camera (Value 800, Weight 2.0)  
  Smartwatch (Value 250, Weight 0.3)  
  Tablet (Value 600, Weight 1.5)  
  Bluetooth Speaker (Value 150, Weight 1.0)  
  Keyboard (Value 80, Weight 0.8)  
  Mouse (Value 50, Weight 0.2)  
  Monitor (Value 300, Weight 3.0)  
  Printer (Value 400, Weight 4.0)  
  External Hard Drive (Value 120, Weight 0.5)  
  Router (Value 60, Weight 0.3)  
  Power Bank (Value 40, Weight 0.4)  
  Memory Card (Value 30, Weight 0.05)  
  USB Flash Drive (Value 25, Weight 0.02)  
  Smart Home Hub (Value 100, Weight 0.6)  
  Gaming Console (Value 500, Weight 4.0)  
  Fitness Tracker (Value 90, Weight 0.2)  
  E-Reader (Value 180, Weight 0.5)

---

**Mathematical Model**

**Decision Variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

**Objective:**  
Maximize total value across all shelves:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

**Constraints:**

For each shelf $i$ (with Capacity $C_i$):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Explicit Data Used:**

- Shelf Capacities (by ShelfID):

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

- Products (by ProductName):

| ProductName           | Value | Weight |
|----------------------|-------|--------|
| Smartphone           | 200   | 1.0    |
| Laptop               | 1500  | 5.0    |
| Headphones           | 100   | 0.5    |
| Camera               | 800   | 2.0    |
| Smartwatch           | 250   | 0.3    |
| Tablet               | 600   | 1.5    |
| Bluetooth Speaker    | 150   | 1.0    |
| Keyboard             | 80    | 0.8    |
| Mouse                | 50    | 0.2    |
| Monitor              | 300   | 3.0    |
| Printer              | 400   | 4.0    |
| External Hard Drive  | 120   | 0.5    |
| Router               | 60    | 0.3    |
| Power Bank           | 40    | 0.4    |
| Memory Card          | 30    | 0.05   |
| USB Flash Drive      | 25    | 0.02   |
| Smart Home Hub       | 100   | 0.6    |
| Gaming Console       | 500   | 4.0    |
| Fitness Tracker      | 90    | 0.2    |
| E-Reader             | 180   | 0.5    |

---

**Summary:**

Maximize
$$
\sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
$$

Subject to, for each shelf $i$:
$$
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$
for all $i=1,\ldots,10$, $j=1,\ldots,20$.