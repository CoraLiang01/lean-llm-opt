**Mathematical Optimization Model**

**Sets and Indices:**
- Let $I$ be the set of cabinets, indexed by $i$ (with CabinetID from capacity.csv).
- Let $J$ be the set of coffee products, indexed by $j$ (with ProductName from products.csv).

**Parameters:**
- $C_i$: Capacity of cabinet $i$ (from capacity.csv).
- $v_j$: Value per unit of product $j$ (from products.csv).
- $w_j$: Weight per unit of product $j$ (from products.csv).

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ to place in cabinet $i$.
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

---

### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

That is, maximize the total value of all coffee products allocated across all cabinets.

---

### Constraints

**1. Cabinet Capacity Constraints**

For each cabinet $i$:

\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

**2. Nonnegativity and Integrality**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

### Data

**Cabinets (from capacity.csv):**

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

**Products (from products.csv):**

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Espresso Beans      | 100   | 1.0    |
| Colombian Roast     | 150   | 1.5    |
| Arabica Blend       | 80    | 1.2    |
| French Roast        | 120   | 1.3    |
| Italian Roast       | 130   | 1.4    |
| House Blend         | 110   | 1.1    |
| Sumatra Coffee      | 160   | 1.8    |
| Mocha Java          | 90    | 1.2    |
| Hazelnut Flavor     | 95    | 1.0    |
| Caramel Blend       | 105   | 1.3    |
| Vanilla Flavor      | 85    | 1.2    |
| Cappuccino Mix      | 140   | 1.5    |
| Pumpkin Spice       | 75    | 1.1    |
| Decaf Roast         | 60    | 1.0    |
| Organic Roast       | 170   | 1.6    |
| Cold Brew           | 115   | 1.4    |
| Peruvian Blend      | 155   | 1.7    |
| Kenyan AA           | 125   | 1.3    |

---

### Complete Model (Numerical Form)

**Variables:**
- $x_{ij}$: Number of units of product $j$ (as listed above) to place in cabinet $i$ (CabinetID $1$ to $10$), $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Objective:**

\[
\max \sum_{i=1}^{10} \Bigg[
100\,x_{i,\text{Espresso Beans}} + 150\,x_{i,\text{Colombian Roast}} + 80\,x_{i,\text{Arabica Blend}} + 120\,x_{i,\text{French Roast}} + 130\,x_{i,\text{Italian Roast}} + 110\,x_{i,\text{House Blend}} + 160\,x_{i,\text{Sumatra Coffee}} + 90\,x_{i,\text{Mocha Java}} + 95\,x_{i,\text{Hazelnut Flavor}} + 105\,x_{i,\text{Caramel Blend}} + 85\,x_{i,\text{Vanilla Flavor}} + 140\,x_{i,\text{Cappuccino Mix}} + 75\,x_{i,\text{Pumpkin Spice}} + 60\,x_{i,\text{Decaf Roast}} + 170\,x_{i,\text{Organic Roast}} + 115\,x_{i,\text{Cold Brew}} + 155\,x_{i,\text{Peruvian Blend}} + 125\,x_{i,\text{Kenyan AA}}
\Bigg]
\]

**Subject to, for each cabinet $i$ ($i=1,\ldots,10$):**

\[
\begin{align*}
&1.0\,x_{i,\text{Espresso Beans}} + 1.5\,x_{i,\text{Colombian Roast}} + 1.2\,x_{i,\text{Arabica Blend}} + 1.3\,x_{i,\text{French Roast}} + 1.4\,x_{i,\text{Italian Roast}} + 1.1\,x_{i,\text{House Blend}} + 1.8\,x_{i,\text{Sumatra Coffee}} + 1.2\,x_{i,\text{Mocha Java}} \\
&\quad + 1.0\,x_{i,\text{Hazelnut Flavor}} + 1.3\,x_{i,\text{Caramel Blend}} + 1.2\,x_{i,\text{Vanilla Flavor}} + 1.5\,x_{i,\text{Cappuccino Mix}} + 1.1\,x_{i,\text{Pumpkin Spice}} + 1.0\,x_{i,\text{Decaf Roast}} \\
&\quad + 1.6\,x_{i,\text{Organic Roast}} + 1.4\,x_{i,\text{Cold Brew}} + 1.7\,x_{i,\text{Peruvian Blend}} + 1.3\,x_{i,\text{Kenyan AA}} \leq C_i
\end{align*}
\]
where $C_i$ is the capacity for cabinet $i$ as listed above.

**And for all $i, j$:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]