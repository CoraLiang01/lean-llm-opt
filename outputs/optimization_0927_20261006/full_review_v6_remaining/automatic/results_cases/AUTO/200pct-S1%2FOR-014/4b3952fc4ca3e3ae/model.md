Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of product $j$ (from Value in products.csv)
- $W_j$: Weight of product $j$ (from Weight in products.csv)
- $C_i$: Capacity of shelf $i$ (from Capacity in capacity.csv)

**Sets:**

- Shelves $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (from ShelfID in capacity.csv)
- Products $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ (from ProductName in products.csv)

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Products}} V_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (using the original order and identifiers):

1. **Shelf Capacity Constraints:**

For $i=1$ (ShelfID=1, Capacity=5.0):
$$
\sum_{j} W_j \cdot x_{1j} \leq 5.0
$$

For $i=2$ (ShelfID=2, Capacity=7.0):
$$
\sum_{j} W_j \cdot x_{2j} \leq 7.0
$$

For $i=3$ (ShelfID=3, Capacity=6.0):
$$
\sum_{j} W_j \cdot x_{3j} \leq 6.0
$$

For $i=4$ (ShelfID=4, Capacity=8.0):
$$
\sum_{j} W_j \cdot x_{4j} \leq 8.0
$$

For $i=5$ (ShelfID=5, Capacity=5.5):
$$
\sum_{j} W_j \cdot x_{5j} \leq 5.5
$$

For $i=6$ (ShelfID=6, Capacity=9.0):
$$
\sum_{j} W_j \cdot x_{6j} \leq 9.0
$$

For $i=7$ (ShelfID=7, Capacity=6.5):
$$
\sum_{j} W_j \cdot x_{7j} \leq 6.5
$$

For $i=8$ (ShelfID=8, Capacity=7.5):
$$
\sum_{j} W_j \cdot x_{8j} \leq 7.5
$$

For $i=9$ (ShelfID=9, Capacity=8.2):
$$
\sum_{j} W_j \cdot x_{9j} \leq 8.2
$$

For $i=10$ (ShelfID=10, Capacity=5.7):
$$
\sum_{j} W_j \cdot x_{10j} \leq 5.7
$$

2. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \text{Products}
$$

**Parameter Table (from products.csv, in source order):**

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

**Parameter Table (from capacity.csv, in source order):**

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

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} V_j x_{ij} \\
\text{s.t. } & \sum_{j=1}^{20} W_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
$$

Where $V_j$, $W_j$, $C_i$ are as specified above, and the product and shelf indices follow the original file order and identifiers.