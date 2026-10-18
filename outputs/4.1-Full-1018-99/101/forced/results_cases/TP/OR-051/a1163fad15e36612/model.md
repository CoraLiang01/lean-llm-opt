##### Sets

- $I = \{1,2,3,4,5,6,7,8,9,10\}$: set of cabinets (CabinetID, in source order)
- $J = \{$Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA$\}$: set of coffee products (ProductName, in source order)

##### Parameters

- $C_i$: capacity of cabinet $i$
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
- $v_j$: value per unit of product $j$
    - Espresso Beans: $v_1 = 100$
    - Colombian Roast: $v_2 = 150$
    - Arabica Blend: $v_3 = 80$
    - French Roast: $v_4 = 120$
    - Italian Roast: $v_5 = 130$
    - House Blend: $v_6 = 110$
    - Sumatra Coffee: $v_7 = 160$
    - Mocha Java: $v_8 = 90$
    - Hazelnut Flavor: $v_9 = 95$
    - Caramel Blend: $v_{10} = 105$
    - Vanilla Flavor: $v_{11} = 85$
    - Cappuccino Mix: $v_{12} = 140$
    - Pumpkin Spice: $v_{13} = 75$
    - Decaf Roast: $v_{14} = 60$
    - Organic Roast: $v_{15} = 170$
    - Cold Brew: $v_{16} = 115$
    - Peruvian Blend: $v_{17} = 155$
    - Kenyan AA: $v_{18} = 125$
- $w_j$: weight per unit of product $j$
    - Espresso Beans: $w_1 = 1.0$
    - Colombian Roast: $w_2 = 1.5$
    - Arabica Blend: $w_3 = 1.2$
    - French Roast: $w_4 = 1.3$
    - Italian Roast: $w_5 = 1.4$
    - House Blend: $w_6 = 1.1$
    - Sumatra Coffee: $w_7 = 1.8$
    - Mocha Java: $w_8 = 1.2$
    - Hazelnut Flavor: $w_9 = 1.0$
    - Caramel Blend: $w_{10} = 1.3$
    - Vanilla Flavor: $w_{11} = 1.2$
    - Cappuccino Mix: $w_{12} = 1.5$
    - Pumpkin Spice: $w_{13} = 1.1$
    - Decaf Roast: $w_{14} = 1.0$
    - Organic Roast: $w_{15} = 1.6$
    - Cold Brew: $w_{16} = 1.4$
    - Peruvian Blend: $w_{17} = 1.7$
    - Kenyan AA: $w_{18} = 1.3$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in cabinet $i$, for all $i \in I$, $j \in J$

##### Objective

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

##### Constraints

1. Capacity constraints for each cabinet:
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
   $$
2. Nonnegativity and integrality:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

##### Source Data

- Cabinets (in source order):

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

- Products (in source order):

| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Espresso Beans     | 100   | 1.0    |
| Colombian Roast    | 150   | 1.5    |
| Arabica Blend      | 80    | 1.2    |
| French Roast       | 120   | 1.3    |
| Italian Roast      | 130   | 1.4    |
| House Blend        | 110   | 1.1    |
| Sumatra Coffee     | 160   | 1.8    |
| Mocha Java         | 90    | 1.2    |
| Hazelnut Flavor    | 95    | 1.0    |
| Caramel Blend      | 105   | 1.3    |
| Vanilla Flavor     | 85    | 1.2    |
| Cappuccino Mix     | 140   | 1.5    |
| Pumpkin Spice      | 75    | 1.1    |
| Decaf Roast        | 60    | 1.0    |
| Organic Roast      | 170   | 1.6    |
| Cold Brew          | 115   | 1.4    |
| Peruvian Blend     | 155   | 1.7    |
| Kenyan AA          | 125   | 1.3    |