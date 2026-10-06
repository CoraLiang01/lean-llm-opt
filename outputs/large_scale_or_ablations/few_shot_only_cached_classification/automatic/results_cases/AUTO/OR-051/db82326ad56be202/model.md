Let $x_{ij}$ = number of units of coffee product $j$ to be placed in cabinet $i$.

Indices:
- $i$ indexes cabinets: $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (CabinetID from capacity.csv, in source order)
- $j$ indexes products, in the order from products.csv:
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

Parameters:
- $v_j$ = value of product $j$ (see table below)
- $w_j$ = weight of product $j$ (see table below)
- $C_i$ = capacity of cabinet $i$ (see table below)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]

Subject to (for all $i = 1,\ldots,10$):
\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
\]

Parameter Table (source order):

| $j$ | ProductName        | $v_j$ | $w_j$ |
|-----|--------------------|-------|-------|
| 1   | Espresso Beans     | 100   | 1.0   |
| 2   | Colombian Roast    | 150   | 1.5   |
| 3   | Arabica Blend      | 80    | 1.2   |
| 4   | French Roast       | 120   | 1.3   |
| 5   | Italian Roast      | 130   | 1.4   |
| 6   | House Blend        | 110   | 1.1   |
| 7   | Sumatra Coffee    | 160   | 1.8   |
| 8   | Mocha Java         | 90    | 1.2   |
| 9   | Hazelnut Flavor    | 95    | 1.0   |
| 10  | Caramel Blend      | 105   | 1.3   |
| 11  | Vanilla Flavor     | 85    | 1.2   |
| 12  | Cappuccino Mix     | 140   | 1.5   |
| 13  | Pumpkin Spice      | 75    | 1.1   |
| 14  | Decaf Roast        | 60    | 1.0   |
| 15  | Organic Roast      | 170   | 1.6   |
| 16  | Cold Brew          | 115   | 1.4   |
| 17  | Peruvian Blend     | 155   | 1.7   |
| 18  | Kenyan AA          | 125   | 1.3   |

Cabinet Capacities (source order):

| $i$ | CabinetID | $C_i$ |
|-----|-----------|-------|
| 1   | 1         | 400   |
| 2   | 2         | 600   |
| 3   | 3         | 500   |
| 4   | 4         | 700   |
| 5   | 5         | 450   |
| 6   | 6         | 650   |
| 7   | 7         | 550   |
| 8   | 8         | 750   |
| 9   | 9         | 480   |
| 10  | 10        | 520   |