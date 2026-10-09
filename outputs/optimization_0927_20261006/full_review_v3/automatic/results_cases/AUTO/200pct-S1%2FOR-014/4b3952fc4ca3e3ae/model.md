Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

Define:
- Shelves $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with capacities $C_i$ as below.
- Products $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$, with values $v_j$ and weights $w_j$ as below.

#### Parameters

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

| Product Name           | Value | Weight |
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

#### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$ as given above.

**Shelf Capacity Constraints:**
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where $w_j$ is the weight of product $j$ and $C_i$ is the capacity of shelf $i$ as given above.

**Integrality and Nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
\]

**Explicit Data Used:**

- Shelf capacities:
    - Shelf 1: 5.0
    - Shelf 2: 7.0
    - Shelf 3: 6.0
    - Shelf 4: 8.0
    - Shelf 5: 5.5
    - Shelf 6: 9.0
    - Shelf 7: 6.5
    - Shelf 8: 7.5
    - Shelf 9: 8.2
    - Shelf 10: 5.7

- Product values and weights:
    - Smartphone: 200, 1.0
    - Laptop: 1500, 5.0
    - Headphones: 100, 0.5
    - Camera: 800, 2.0
    - Smartwatch: 250, 0.3
    - Tablet: 600, 1.5
    - Bluetooth Speaker: 150, 1.0
    - Keyboard: 80, 0.8
    - Mouse: 50, 0.2
    - Monitor: 300, 3.0
    - Printer: 400, 4.0
    - External Hard Drive: 120, 0.5
    - Router: 60, 0.3
    - Power Bank: 40, 0.4
    - Memory Card: 30, 0.05
    - USB Flash Drive: 25, 0.02
    - Smart Home Hub: 100, 0.6
    - Gaming Console: 500, 4.0
    - Fitness Tracker: 90, 0.2
    - E-Reader: 180, 0.5

**Decision variables:**
- $x_{ij}$: Number of units of product $j$ placed on shelf $i$, integer and nonnegative.

**Summary:**
Maximize total value of products allocated to shelves, subject to each shelf's weight capacity, using integer allocations per product per shelf. All data and identifiers are preserved as retrieved.