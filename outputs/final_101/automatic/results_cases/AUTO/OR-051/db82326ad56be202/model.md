Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i$ indexes cabinets: $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (CabinetID)
- $j$ indexes products, in the order below.

Parameters:
- $c_i$ = capacity of cabinet $i$ (see below)
- $v_j$ = value per unit of product $j$ (see below)
- $w_j$ = weight per unit of product $j$ (see below)

Cabinet capacities:
\[
\begin{align*}
c_1 &= 400 \\
c_2 &= 600 \\
c_3 &= 500 \\
c_4 &= 700 \\
c_5 &= 450 \\
c_6 &= 650 \\
c_7 &= 550 \\
c_8 &= 750 \\
c_9 &= 480 \\
c_{10} &= 520 \\
\end{align*}
\]

Products (in source order):

| $j$ | ProductName         | $v_j$ | $w_j$ |
|-----|---------------------|-------|-------|
| 1   | Espresso Beans      | 100   | 1.0   |
| 2   | Colombian Roast     | 150   | 1.5   |
| 3   | Arabica Blend      | 80    | 1.2   |
| 4   | French Roast        | 120   | 1.3   |
| 5   | Italian Roast       | 130   | 1.4   |
| 6   | House Blend         | 110   | 1.1   |
| 7   | Sumatra Coffee      | 160   | 1.8   |
| 8   | Mocha Java          | 90    | 1.2   |
| 9   | Hazelnut Flavor     | 95    | 1.0   |
| 10  | Caramel Blend       | 105   | 1.3   |
| 11  | Vanilla Flavor      | 85    | 1.2   |
| 12  | Cappuccino Mix      | 140   | 1.5   |
| 13  | Pumpkin Spice       | 75    | 1.1   |
| 14  | Decaf Roast         | 60    | 1.0   |
| 15  | Organic Roast       | 170   | 1.6   |
| 16  | Cold Brew           | 115   | 1.4   |
| 17  | Peruvian Blend      | 155   | 1.7   |
| 18  | Kenyan AA           | 125   | 1.3   |

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]

Subject to, for each cabinet $i$:
\[
\sum_{j=1}^{18} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,18
\]