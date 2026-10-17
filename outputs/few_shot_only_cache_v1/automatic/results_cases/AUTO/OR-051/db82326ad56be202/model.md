Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are nonnegative integers.

Let:
- $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$ index cabinets, with CabinetID as below.
- $j \in \{\text{Espresso Beans}, \text{Colombian Roast}, \text{Arabica Blend}, \text{French Roast}, \text{Italian Roast}, \text{House Blend}, \text{Sumatra Coffee}, \text{Mocha Java}, \text{Hazelnut Flavor}, \text{Caramel Blend}, \text{Vanilla Flavor}, \text{Cappuccino Mix}, \text{Pumpkin Spice}, \text{Decaf Roast}, \text{Organic Roast}, \text{Cold Brew}, \text{Peruvian Blend}, \text{Kenyan AA}\}$ index products.

Let $v_j$ be the value of product $j$ and $w_j$ its weight per unit, as given below.

Let $C_i$ be the capacity of cabinet $i$.

#### Data

Cabinets and capacities:
- Cabinet 1: $C_1 = 400$
- Cabinet 2: $C_2 = 600$
- Cabinet 3: $C_3 = 500$
- Cabinet 4: $C_4 = 700$
- Cabinet 5: $C_5 = 450$
- Cabinet 6: $C_6 = 650$
- Cabinet 7: $C_7 = 550$
- Cabinet 8: $C_8 = 750$
- Cabinet 9: $C_9 = 480$
- Cabinet 10: $C_{10} = 520$

Products, values, and weights:
- Espresso Beans: $v = 100$, $w = 1.0$
- Colombian Roast: $v = 150$, $w = 1.5$
- Arabica Blend: $v = 80$, $w = 1.2$
- French Roast: $v = 120$, $w = 1.3$
- Italian Roast: $v = 130$, $w = 1.4$
- House Blend: $v = 110$, $w = 1.1$
- Sumatra Coffee: $v = 160$, $w = 1.8$
- Mocha Java: $v = 90$, $w = 1.2$
- Hazelnut Flavor: $v = 95$, $w = 1.0$
- Caramel Blend: $v = 105$, $w = 1.3$
- Vanilla Flavor: $v = 85$, $w = 1.2$
- Cappuccino Mix: $v = 140$, $w = 1.5$
- Pumpkin Spice: $v = 75$, $w = 1.1$
- Decaf Roast: $v = 60$, $w = 1.0$
- Organic Roast: $v = 170$, $w = 1.6$
- Cold Brew: $v = 115$, $w = 1.4$
- Peruvian Blend: $v = 155$, $w = 1.7$
- Kenyan AA: $v = 125$, $w = 1.3$

#### Mathematical Model

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]

Subject to, for each cabinet $i$:
\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; \forall j \in \{\text{all 18 products as listed above}\}
\]

Where:
- $v_j$ and $w_j$ are as listed above for each product $j$,
- $C_i$ is as listed above for each cabinet $i$.