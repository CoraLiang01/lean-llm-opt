##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of coffee product $j$ placed in cabinet $i$, for each cabinet $i \in I$ and product $j \in J$.

##### Parameters

- Cabinets $I = \{1,2,3,4,5,6,7,8,9,10\}$
- Cabinet capacities:
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

- Products $J = \{$
  1: Espresso Beans,
  2: Colombian Roast,
  3: Arabica Blend,
  4: French Roast,
  5: Italian Roast,
  6: House Blend,
  7: Sumatra Coffee,
  8: Mocha Java,
  9: Hazelnut Flavor,
  10: Caramel Blend,
  11: Vanilla Flavor,
  12: Cappuccino Mix,
  13: Pumpkin Spice,
  14: Decaf Roast,
  15: Organic Roast,
  16: Cold Brew,
  17: Peruvian Blend,
  18: Kenyan AA
$\}$

- Product values $v_j$ and weights $w_j$:

| $j$ | Product Name        | $v_j$ | $w_j$ |
|-----|--------------------|-------|-------|
| 1   | Espresso Beans     | 100   | 1.0   |
| 2   | Colombian Roast    | 150   | 1.5   |
| 3   | Arabica Blend      | 80    | 1.2   |
| 4   | French Roast       | 120   | 1.3   |
| 5   | Italian Roast      | 130   | 1.4   |
| 6   | House Blend        | 110   | 1.1   |
| 7   | Sumatra Coffee     | 160   | 1.8   |
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

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Cabinet capacity constraints:** For each cabinet $i \in I$,
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i
   \]
   where $C_i$ is the capacity of cabinet $i$.

2. **Integrality constraints:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

##### All parameters (from CSV):

- Cabinets and capacities:
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

- Products, values, and weights:
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