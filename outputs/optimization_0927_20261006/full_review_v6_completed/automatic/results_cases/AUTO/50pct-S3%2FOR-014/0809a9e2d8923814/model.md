Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

Define:
- Shelves $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with ShelfID as given below.
- Products $j$ as listed below, with their Value and Weight.

#### Sets and Parameters

**Shelves (from capacity.csv):**
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

**Products (from products.csv):**
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

#### Mathematical Model

**Decision Variables:**
$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

**Shelf Capacity Constraints:**
For each shelf $i$ (with Capacity $C_i$):
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$
where $w_j$ is the Weight of product $j$.

**Integrality and Nonnegativity:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ \forall j \in \{1,2,\ldots,20\}
$$

#### Explicit Data Mapping

- Shelf indices $i$ correspond to ShelfID as listed above.
- Product indices $j$ correspond to the order in the products table above.
- $v_j$ and $w_j$ are the Value and Weight for product $j$ as listed.

This model maximizes the total value of products allocated to shelves, subject to each shelf's capacity, with integer, nonnegative allocations.