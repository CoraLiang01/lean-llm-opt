Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index cabinets, with CabinetID as below.
- $j$ index products, with ProductName as below.
- $c_i$ = Capacity of cabinet $i$.
- $v_j$ = Value of product $j$.
- $w_j$ = Weight of product $j$.

#### Sets and Parameters (from retrieved data):

Cabinets (CabinetID, Capacity):
- 1: 400
- 2: 600
- 3: 500
- 4: 700
- 5: 450
- 6: 650
- 7: 550
- 8: 750
- 9: 480
- 10: 520

Products (ProductName, Value, Weight):
- Espresso Beans: 100, 1.0
- Colombian Roast: 150, 1.5
- Arabica Blend: 80, 1.2
- French Roast: 120, 1.3
- Italian Roast: 130, 1.4
- House Blend: 110, 1.1
- Sumatra Coffee: 160, 1.8
- Mocha Java: 90, 1.2
- Hazelnut Flavor: 95, 1.0
- Caramel Blend: 105, 1.3
- Vanilla Flavor: 85, 1.2
- Cappuccino Mix: 140, 1.5
- Pumpkin Spice: 75, 1.1
- Decaf Roast: 60, 1.0
- Organic Roast: 170, 1.6
- Cold Brew: 115, 1.4
- Peruvian Blend: 155, 1.7
- Kenyan AA: 125, 1.3

#### Decision Variables

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all products above}\}
$$

#### Objective Function

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij}
$$

where $v_j$ is the value of product $j$ as listed above.

#### Constraints

For each cabinet $i$ (CabinetID):

$$
\sum_{j=1}^{18} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the weight of product $j$ and $c_i$ is the capacity of cabinet $i$ as listed above.

#### Variable Domains

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

#### Explicit Data Table

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

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{18} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,18
\end{align*}
$$

where $v_j$, $w_j$, and $c_i$ are as listed above, and $x_{ij}$ is the integer number of units of product $j$ in cabinet $i$.