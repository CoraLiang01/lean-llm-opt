Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are nonnegative integers.

Define:
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (CabinetID, in source order)
- $j \in \{$Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA$\}$ (ProductName, in source order)

Let $v_j$ be the value of product $j$ and $w_j$ its weight, as below:

| ProductName         | $v_j$ | $w_j$ |
|---------------------|-------|-------|
| Espresso Beans      | 100   | 1.0   |
| Colombian Roast     | 150   | 1.5   |
| Arabica Blend       | 80    | 1.2   |
| French Roast        | 120   | 1.3   |
| Italian Roast       | 130   | 1.4   |
| House Blend         | 110   | 1.1   |
| Sumatra Coffee      | 160   | 1.8   |
| Mocha Java          | 90    | 1.2   |
| Hazelnut Flavor     | 95    | 1.0   |
| Caramel Blend       | 105   | 1.3   |
| Vanilla Flavor      | 85    | 1.2   |
| Cappuccino Mix      | 140   | 1.5   |
| Pumpkin Spice       | 75    | 1.1   |
| Decaf Roast         | 60    | 1.0   |
| Organic Roast       | 170   | 1.6   |
| Cold Brew           | 115   | 1.4   |
| Peruvian Blend      | 155   | 1.7   |
| Kenyan AA           | 125   | 1.3   |

Let $C_i$ be the capacity of cabinet $i$:

| CabinetID | $C_i$ |
|-----------|-------|
| 1         | 400   |
| 2         | 600   |
| 3         | 500   |
| 4         | 700   |
| 5         | 450   |
| 6         | 650   |
| 7         | 550   |
| 8         | 750   |
| 9         | 480   |
| 10        | 520   |

The mathematical model is:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
$$

Subject to, for each cabinet $i$:
$$
\sum_{j=1}^{18} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; \forall j \in \{1,2,\ldots,18\}
$$

Where the mapping of $j$ to ProductName, $v_j$, and $w_j$ is as listed above, and the mapping of $i$ to CabinetID and $C_i$ is as listed above.