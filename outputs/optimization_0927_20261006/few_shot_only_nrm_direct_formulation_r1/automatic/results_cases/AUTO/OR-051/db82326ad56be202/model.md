**Sets:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (CabinetID from "capacity.csv")
- $j \in \{$Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA$\}$ (ProductName from "products.csv")

**Parameters:**
- $C_i$ = Capacity of cabinet $i$ (from "capacity.csv")
- $v_j$ = Value of product $j$ (from "products.csv")
- $w_j$ = Weight of product $j$ (from "products.csv")

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed in cabinet $i$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij}
\]

**Subject to:**

For each cabinet $i$:
\[
\sum_{j=1}^{18} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Data:**

From "capacity.csv":

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

From "products.csv":

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
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{18} w_j \cdot x_{i j} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,18
\end{align*}
\]

Where:

- $C_i$ is the Capacity for CabinetID $i$ as above.
- $v_j$ and $w_j$ are the Value and Weight for Product $j$ as above.
- $x_{ij}$ is the integer number of units of product $j$ in cabinet $i$.