##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of cabinets, indexed by $i$.

Let $J = \{$Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA$\}$ be the set of coffee products, indexed by $j$.

##### Parameters

Cabinet capacities:
\[
\begin{align*}
C_1 &= 400 \\
C_2 &= 600 \\
C_3 &= 500 \\
C_4 &= 700 \\
C_5 &= 450 \\
C_6 &= 650 \\
C_7 &= 550 \\
C_8 &= 750 \\
C_9 &= 480 \\
C_{10} &= 520 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{lll}
\text{Product} & v_j & w_j \\
\hline
\text{Espresso Beans} & 100 & 1.0 \\
\text{Colombian Roast} & 150 & 1.5 \\
\text{Arabica Blend} & 80 & 1.2 \\
\text{French Roast} & 120 & 1.3 \\
\text{Italian Roast} & 130 & 1.4 \\
\text{House Blend} & 110 & 1.1 \\
\text{Sumatra Coffee} & 160 & 1.8 \\
\text{Mocha Java} & 90 & 1.2 \\
\text{Hazelnut Flavor} & 95 & 1.0 \\
\text{Caramel Blend} & 105 & 1.3 \\
\text{Vanilla Flavor} & 85 & 1.2 \\
\text{Cappuccino Mix} & 140 & 1.5 \\
\text{Pumpkin Spice} & 75 & 1.1 \\
\text{Decaf Roast} & 60 & 1.0 \\
\text{Organic Roast} & 170 & 1.6 \\
\text{Cold Brew} & 115 & 1.4 \\
\text{Peruvian Blend} & 155 & 1.7 \\
\text{Kenyan AA} & 125 & 1.3 \\
\end{array}
\]

##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed in cabinet $i$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Cabinet capacity constraints:** For each cabinet $i \in I$,
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i
   \]
2. **Integrality:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

##### Full Parameter Listing

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
  - Espresso Beans: value 100, weight 1.0
  - Colombian Roast: value 150, weight 1.5
  - Arabica Blend: value 80, weight 1.2
  - French Roast: value 120, weight 1.3
  - Italian Roast: value 130, weight 1.4
  - House Blend: value 110, weight 1.1
  - Sumatra Coffee: value 160, weight 1.8
  - Mocha Java: value 90, weight 1.2
  - Hazelnut Flavor: value 95, weight 1.0
  - Caramel Blend: value 105, weight 1.3
  - Vanilla Flavor: value 85, weight 1.2
  - Cappuccino Mix: value 140, weight 1.5
  - Pumpkin Spice: value 75, weight 1.1
  - Decaf Roast: value 60, weight 1.0
  - Organic Roast: value 170, weight 1.6
  - Cold Brew: value 115, weight 1.4
  - Peruvian Blend: value 155, weight 1.7
  - Kenyan AA: value 125, weight 1.3