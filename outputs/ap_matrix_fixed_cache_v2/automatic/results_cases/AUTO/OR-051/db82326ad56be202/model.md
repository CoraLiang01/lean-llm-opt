Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes cabinets, with CabinetID from 1 to 10 (see below).
- $j$ indexes coffee products, with ProductName as below.

**Parameters:**
- $v_j$: Value of one unit of product $j$ (from products.csv).
- $w_j$: Weight of one unit of product $j$ (from products.csv).
- $C_i$: Capacity of cabinet $i$ (from capacity.csv).

**Data:**

Cabinets (from capacity.csv, in source order):

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

Coffee Products (from products.csv, in source order):

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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
\]

Subject to, for each cabinet $i$ (CabinetID as above):
\[
\sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ \forall j \in \text{Products}
\]

Where:
- $\text{Products}$ is the set of 18 coffee products listed above, with their Value and Weight as given.

**All coefficients and identifiers are as retrieved and in original order.**