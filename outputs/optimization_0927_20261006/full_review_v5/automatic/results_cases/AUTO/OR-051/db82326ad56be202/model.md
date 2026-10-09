Let $x_{ij}$ be the number of units of coffee product $j$ placed in cabinet $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i$ indexes cabinets, with CabinetID as below.
- $j$ indexes coffee products, with ProductName as below.

**Parameters:**

- $C_i$: Capacity of cabinet $i$ (from Capacity column).
- $v_j$: Value per unit of product $j$ (from Value column).
- $w_j$: Weight per unit of product $j$ (from Weight column).

---

### Objective

$$
\max \sum_{i} \sum_{j} v_j \cdot x_{ij}
$$

---

### Constraints

**Cabinet Capacity Constraints:**

For each cabinet $i$:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i
$$

**Integrality and Nonnegativity:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

---

### Data

#### Cabinets (from capacity.csv, in source order):

| CabinetID | Capacity |
|-----------|----------|
| 1         | 400      |
| 2         | 600      |
| 3         | 500      |
| 4         | 700      |
| 5         | 450      |
| 6         | 650      |
| 7         | 550      |
| 8         | 750      |
| 9         | 480      |
| 10        | 520      |

#### Coffee Products (from products.csv, in source order):

| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Espresso Beans     | 100   | 1.0    |
| Colombian Roast    | 150   | 1.5    |
| Arabica Blend      | 80    | 1.2    |
| French Roast       | 120   | 1.3    |
| Italian Roast      | 130   | 1.4    |
| House Blend        | 110   | 1.1    |
| Sumatra Coffee     | 160   | 1.8    |
| Mocha Java         | 90    | 1.2    |
| Hazelnut Flavor    | 95    | 1.0    |
| Caramel Blend      | 105   | 1.3    |
| Vanilla Flavor     | 85    | 1.2    |
| Cappuccino Mix     | 140   | 1.5    |
| Pumpkin Spice      | 75    | 1.1    |
| Decaf Roast        | 60    | 1.0    |
| Organic Roast      | 170   | 1.6    |
| Cold Brew          | 115   | 1.4    |
| Peruvian Blend     | 155   | 1.7    |
| Kenyan AA          | 125   | 1.3    |

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,18\}} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{18} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\end{align*}
$$

Where the mapping from $i$ to CabinetID and $j$ to ProductName, $v_j$, $w_j$ is as given in the tables above.