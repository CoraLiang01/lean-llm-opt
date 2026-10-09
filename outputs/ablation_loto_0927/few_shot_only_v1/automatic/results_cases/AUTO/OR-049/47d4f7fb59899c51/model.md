Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are required to be nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, indexed by $i$ (with ShelfID as below).
- Let $P$ be the set of products, indexed by $j$ (with ProductName as below).
- $c_i$ = Capacity of shelf $i$ (from capacity.csv).
- $v_j$ = Value of product $j$ (from products.csv).
- $w_j$ = Weight of product $j$ (from products.csv).

**Data:**

- Shelves ($S$) and their capacities ($c_i$):

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

- Products ($P$) and their values ($v_j$) and weights ($w_j$):

| ProductName            | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1.0    |
| Laptop                 | 1500  | 5.0    |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2.0    |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1.0    |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3.0    |
| Printer                | 400   | 4.0    |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4.0    |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad \forall i \in S,\, j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:**  
   For each shelf $i \in S$,
   $$
   \sum_{j \in P} w_j \, x_{ij} \leq c_i
   $$

2. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   $$

---

**Where:**

- $S = \{1,2,3,4,5,6,7,8,9,10\}$
- $P =$  
  $\{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$
- $c_i$ as given above for each $i$
- $v_j$, $w_j$ as given above for each $j$