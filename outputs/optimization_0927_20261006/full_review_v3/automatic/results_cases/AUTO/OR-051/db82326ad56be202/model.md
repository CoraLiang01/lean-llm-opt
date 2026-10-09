Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $I$ be the set of cabinets (indexed by CabinetID), and $J$ be the set of coffee products (indexed by ProductName).

Let $c_i$ be the capacity of cabinet $i$.

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Sets and Parameters (from the data):

Cabinets ($i$):

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

Coffee Products ($j$):

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

Mathematical Model:

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to (for all $i \in I$):

Capacity constraints:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Where:

- $I = \{1,2,3,4,5,6,7,8,9,10\}$
- $J =$ {Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA}
- $c_i$ as given above
- $v_j$, $w_j$ as given above

All variables and coefficients are as retrieved and in original order.