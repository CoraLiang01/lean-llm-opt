**Sets:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (CabinetID from capacity.csv)
- $j \in \{\text{Espresso Beans}, \text{Colombian Roast}, \text{Arabica Blend}, \text{French Roast}, \text{Italian Roast}, \text{House Blend}, \text{Sumatra Coffee}, \text{Mocha Java}, \text{Hazelnut Flavor}, \text{Caramel Blend}, \text{Vanilla Flavor}, \text{Cappuccino Mix}, \text{Pumpkin Spice}, \text{Decaf Roast}, \text{Organic Roast}, \text{Cold Brew}, \text{Peruvian Blend}, \text{Kenyan AA}\}$ (ProductName from products.csv)

**Parameters:**
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of cabinet $i$ (from capacity.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed in cabinet $i$

**Objective:**
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Espresso Beans}, \ldots, \text{Kenyan AA}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each cabinet $i$:
\[
\sum_{j \in \{\text{Espresso Beans}, \ldots, \text{Kenyan AA}\}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Parameter Values (from CSVs):**

- Cabinet capacities ($C_i$):

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

- Product values ($v_j$) and weights ($w_j$):

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

**Full Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{18} w_j x_{i1,j} \leq 400 \\
& \sum_{j=1}^{18} w_j x_{i2,j} \leq 600 \\
& \sum_{j=1}^{18} w_j x_{i3,j} \leq 500 \\
& \sum_{j=1}^{18} w_j x_{i4,j} \leq 700 \\
& \sum_{j=1}^{18} w_j x_{i5,j} \leq 450 \\
& \sum_{j=1}^{18} w_j x_{i6,j} \leq 650 \\
& \sum_{j=1}^{18} w_j x_{i7,j} \leq 550 \\
& \sum_{j=1}^{18} w_j x_{i8,j} \leq 750 \\
& \sum_{j=1}^{18} w_j x_{i9,j} \leq 480 \\
& \sum_{j=1}^{18} w_j x_{i10,j} \leq 520 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Espresso Beans}, \ldots, \text{Kenyan AA}\}
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above for each product $j$.