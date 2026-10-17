Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$, where $i$ indexes cabinets and $j$ indexes products. All $x_{ij}$ are integer and $\geq 0$.

Define:
- Cabinets $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with capacities $C_i$:
  - $C_1 = 400$
  - $C_2 = 600$
  - $C_3 = 500$
  - $C_4 = 700$
  - $C_5 = 450$
  - $C_6 = 650$
  - $C_7 = 550$
  - $C_8 = 750$
  - $C_9 = 480$
  - $C_{10} = 520$

- Products $j$ with (Value $v_j$, Weight $w_j$):

| $j$ | ProductName         | $v_j$ | $w_j$ |
|-----|---------------------|-------|-------|
| 1   | Espresso Beans      | 100   | 1.0   |
| 2   | Colombian Roast     | 150   | 1.5   |
| 3   | Arabica Blend       | 80    | 1.2   |
| 4   | French Roast        | 120   | 1.3   |
| 5   | Italian Roast       | 130   | 1.4   |
| 6   | House Blend         | 110   | 1.1   |
| 7   | Sumatra Coffee     | 160   | 1.8   |
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

Mathematical Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]

Subject to, for each cabinet $i$:
\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\]

Where:
- $v_j$ and $w_j$ are as listed above for each product $j$.
- $C_i$ is the capacity of cabinet $i$ as listed above.