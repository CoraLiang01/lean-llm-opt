Let $x_{ij}$ be the number of units of product $j$ (with ProductName as below) placed on shelf $i$ (with ShelfID as below). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv):

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

- Products (from products.csv):

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

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all ProductNames above}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

**Constraints:**

For each shelf $i$ (with ShelfID as above):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Explicitly, for each shelf $i$ (ShelfID $i$):**

$$
\sum_{j} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

with all $x_{ij}$ integer and nonnegative, and the objective as above. All identifiers and coefficients are as in the tables above.