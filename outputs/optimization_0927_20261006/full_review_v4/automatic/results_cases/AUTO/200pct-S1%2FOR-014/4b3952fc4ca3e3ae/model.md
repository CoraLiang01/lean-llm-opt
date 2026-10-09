Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

**Sets and Indices:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv)

**Parameters:**

- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of shelf $i$ (from capacity.csv)

**Objective:**

$$
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$:

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,10\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Parameter Values (from retrieved data):**

- Shelf Capacities:

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

- Product Values and Weights:

  | ProductName            | Value | Weight |
  |----------------------- |-------|--------|
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

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{i1j} \leq 5.0 \\
& \sum_{j=1}^{20} w_j x_{i2j} \leq 7.0 \\
& \sum_{j=1}^{20} w_j x_{i3j} \leq 6.0 \\
& \sum_{j=1}^{20} w_j x_{i4j} \leq 8.0 \\
& \sum_{j=1}^{20} w_j x_{i5j} \leq 5.5 \\
& \sum_{j=1}^{20} w_j x_{i6j} \leq 9.0 \\
& \sum_{j=1}^{20} w_j x_{i7j} \leq 6.5 \\
& \sum_{j=1}^{20} w_j x_{i8j} \leq 7.5 \\
& \sum_{j=1}^{20} w_j x_{i9j} \leq 8.2 \\
& \sum_{j=1}^{20} w_j x_{i10j} \leq 5.7 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
$$

Where $v_j$ and $w_j$ are as listed above, and $x_{ij}$ is the integer number of units of product $j$ on shelf $i$.