##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of coffee product $j$ to be placed in cabinet $i$, for each cabinet $i \in I$ and product $j \in J$.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10\}$ (Cabinet IDs)
- $J = \{$
  Espresso Beans,
  Colombian Roast,
  Arabica Blend,
  French Roast,
  Italian Roast,
  House Blend,
  Sumatra Coffee,
  Mocha Java,
  Hazelnut Flavor,
  Caramel Blend,
  Vanilla Flavor,
  Cappuccino Mix,
  Pumpkin Spice,
  Decaf Roast,
  Organic Roast,
  Cold Brew,
  Peruvian Blend,
  Kenyan AA
$\}$ (Product Names)

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

- Product values and weights:
  - Espresso Beans: $v_1 = 100$, $w_1 = 1.0$
  - Colombian Roast: $v_2 = 150$, $w_2 = 1.5$
  - Arabica Blend: $v_3 = 80$, $w_3 = 1.2$
  - French Roast: $v_4 = 120$, $w_4 = 1.3$
  - Italian Roast: $v_5 = 130$, $w_5 = 1.4$
  - House Blend: $v_6 = 110$, $w_6 = 1.1$
  - Sumatra Coffee: $v_7 = 160$, $w_7 = 1.8$
  - Mocha Java: $v_8 = 90$, $w_8 = 1.2$
  - Hazelnut Flavor: $v_9 = 95$, $w_9 = 1.0$
  - Caramel Blend: $v_{10} = 105$, $w_{10} = 1.3$
  - Vanilla Flavor: $v_{11} = 85$, $w_{11} = 1.2$
  - Cappuccino Mix: $v_{12} = 140$, $w_{12} = 1.5$
  - Pumpkin Spice: $v_{13} = 75$, $w_{13} = 1.1$
  - Decaf Roast: $v_{14} = 60$, $w_{14} = 1.0$
  - Organic Roast: $v_{15} = 170$, $w_{15} = 1.6$
  - Cold Brew: $v_{16} = 115$, $w_{16} = 1.4$
  - Peruvian Blend: $v_{17} = 155$, $w_{17} = 1.7$
  - Kenyan AA: $v_{18} = 125$, $w_{18} = 1.3$

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Cabinet capacity constraints** (for each cabinet $i \in I$):

   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i
   \]

2. **Nonnegativity and integrality** (for all $i \in I$, $j \in J$):

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{18} w_j x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\ j = 1,\ldots,18
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above for each product $j$, and $C_i$ is the capacity of cabinet $i$ as listed above.