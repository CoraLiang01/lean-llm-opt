Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $I$ be the set of displays (shelves), indexed by $i$ and identified by "ShelfID" in the order given:
$$
I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $J$ be the set of products, indexed by $j$ and identified by "ProductName" in the order given:
$$
J = \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}
$$

Let $v_j$ be the value of product $j$ (from "Value" in products.csv).

Let $w_j$ be the weight of product $j$ (from "Weight" in products.csv).

Let $C_i$ be the capacity of display $i$ (from "Capacity" in capacity.csv).

The model is:

---

**Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

---

**Constraints:**

1. **Display Capacity Constraints:** For each display $i$,
   $$
   \sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
   $$
   That is, for each shelf, the total weight of products placed does not exceed its capacity.

2. **Minimum Placement of First Product:** (First product is "Smartphone")
   $$
   \sum_{i \in I} x_{i,\,\text{Smartphone}} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

---

**Parameter Table (in source order):**

*Displays (from capacity.csv):*

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

*Products (from products.csv):*

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

---

**Complete Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j\, x_{ij} \leq C_i \qquad \forall i=1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i,\,\text{Smartphone}} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\; j=1,\ldots,20
\end{align*}
$$

where $v_j$, $w_j$, and $C_i$ are as given in the tables above, and the mapping of $j$ to product names is in source order.