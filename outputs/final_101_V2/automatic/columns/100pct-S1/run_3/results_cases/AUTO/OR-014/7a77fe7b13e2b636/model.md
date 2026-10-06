Let $x_{ij}$ be the number of units of product $j$ (ProductName) placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

    | ShelfID |
    |---------|
    | 1       |
    | 2       |
    | 3       |
    | 4       |
    | 5       |
    | 6       |
    | 7       |
    | 8       |
    | 9       |
    | 10      |

    Shelf capacities:

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

- Products (from products.csv, in order):

    | ProductName           | Value | Weight |
    |---------------------- |-------|--------|
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

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,\ldots,10\}$ (ShelfID), $j$ indexes the 20 products in the order above.

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

---

**Constraints:**

For each shelf $i$ (ShelfID):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of shelf $i$.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly:**

Let $v_j$ and $w_j$ be as in the table above, and $C_i$ as in the shelf table.

For example, for shelf 1:

$$
1.0\,x_{1,1} + 5.0\,x_{1,2} + 0.5\,x_{1,3} + 2.0\,x_{1,4} + 0.3\,x_{1,5} + 1.5\,x_{1,6} + 1.0\,x_{1,7} + 0.8\,x_{1,8} + 0.2\,x_{1,9} + 3.0\,x_{1,10} + 4.0\,x_{1,11} + 0.5\,x_{1,12} + 0.3\,x_{1,13} + 0.4\,x_{1,14} + 0.05\,x_{1,15} + 0.02\,x_{1,16} + 0.6\,x_{1,17} + 4.0\,x_{1,18} + 0.2\,x_{1,19} + 0.5\,x_{1,20} \leq 5.0
$$

And similarly for shelves 2 through 10, using their respective capacities.

---

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Summary of Sets and Parameters:**

- Shelves: $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv, in order)
- Products: $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv, in order)
- $v_j$: Value of product $j$
- $w_j$: Weight of product $j$
- $C_i$: Capacity of shelf $i$

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,20
\end{align*}
$$

with all coefficients and identifiers as above.