Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and nonnegative.

Define:
- Cabinets $i$ (CabinetID): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Products $j$ (ProductName): 
  1. Espresso Beans
  2. Colombian Roast
  3. Arabica Blend
  4. French Roast
  5. Italian Roast
  6. House Blend
  7. Sumatra Coffee
  8. Mocha Java
  9. Hazelnut Flavor
  10. Caramel Blend
  11. Vanilla Flavor
  12. Cappuccino Mix
  13. Pumpkin Spice
  14. Decaf Roast
  15. Organic Roast
  16. Cold Brew
  17. Peruvian Blend
  18. Kenyan AA

Let $v_j$ be the value and $w_j$ the weight of product $j$ (see table below). Let $C_i$ be the capacity of cabinet $i$.

#### Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]

#### Subject to (for each cabinet $i$):

\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\]

#### Data

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

| ProductName         | $v_j$ (Value) | $w_j$ (Weight) |
|---------------------|--------------|---------------|
| Espresso Beans      | 100          | 1.0           |
| Colombian Roast     | 150          | 1.5           |
| Arabica Blend       | 80           | 1.2           |
| French Roast        | 120          | 1.3           |
| Italian Roast       | 130          | 1.4           |
| House Blend         | 110          | 1.1           |
| Sumatra Coffee      | 160          | 1.8           |
| Mocha Java          | 90           | 1.2           |
| Hazelnut Flavor     | 95           | 1.0           |
| Caramel Blend       | 105          | 1.3           |
| Vanilla Flavor      | 85           | 1.2           |
| Cappuccino Mix      | 140          | 1.5           |
| Pumpkin Spice       | 75           | 1.1           |
| Decaf Roast         | 60           | 1.0           |
| Organic Roast       | 170          | 1.6           |
| Cold Brew           | 115          | 1.4           |
| Peruvian Blend      | 155          | 1.7           |
| Kenyan AA           | 125          | 1.3           |