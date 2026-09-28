Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Indices:
- $i$ indexes cabinets, with CabinetID from 1 to 10.
- $j$ indexes coffee products, with ProductName as below.

Parameters:
- $c_i$: Capacity of cabinet $i$ (from "capacity.csv")
- $v_j$: Value per unit of product $j$ (from "products.csv")
- $w_j$: Weight per unit of product $j$ (from "products.csv")

Cabinet capacities:
- $c_1 = 400$
- $c_2 = 600$
- $c_3 = 500$
- $c_4 = 700$
- $c_5 = 450$
- $c_6 = 650$
- $c_7 = 550$
- $c_8 = 750$
- $c_9 = 480$
- $c_{10} = 520$

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

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
$$

Subject to, for each cabinet $i$:
$$
\sum_{j=1}^{18} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

Integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where the mapping of $j$ to ProductName, $v_j$, and $w_j$ is as listed above.