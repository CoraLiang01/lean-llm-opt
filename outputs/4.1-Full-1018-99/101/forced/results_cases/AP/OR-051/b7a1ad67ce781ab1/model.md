##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of cabinets, indexed by $i$.

Let $J = \{$
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
$\}$ be the set of coffee products, indexed by $j$.

##### Parameters

Cabinet capacities:
\[
\begin{align*}
\text{Capacity}_1 &= 400 \\
\text{Capacity}_2 &= 600 \\
\text{Capacity}_3 &= 500 \\
\text{Capacity}_4 &= 700 \\
\text{Capacity}_5 &= 450 \\
\text{Capacity}_6 &= 650 \\
\text{Capacity}_7 &= 550 \\
\text{Capacity}_8 &= 750 \\
\text{Capacity}_9 &= 480 \\
\text{Capacity}_{10} &= 520 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{lll}
\text{Product} & \text{Value} & \text{Weight} \\
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

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of cabinet $i$.

##### Decision Variables

For each cabinet $i \in I$ and product $j \in J$:

\[
x_{ij} = \text{number of units of product } j \text{ placed in cabinet } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each cabinet $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Retrieved Information

{
  "cabinets": {
    "1": 400,
    "2": 600,
    "3": 500,
    "4": 700,
    "5": 450,
    "6": 650,
    "7": 550,
    "8": 750,
    "9": 480,
    "10": 520
  },
  "products": {
    "Espresso Beans": {"Value": 100, "Weight": 1.0},
    "Colombian Roast": {"Value": 150, "Weight": 1.5},
    "Arabica Blend": {"Value": 80, "Weight": 1.2},
    "French Roast": {"Value": 120, "Weight": 1.3},
    "Italian Roast": {"Value": 130, "Weight": 1.4},
    "House Blend": {"Value": 110, "Weight": 1.1},
    "Sumatra Coffee": {"Value": 160, "Weight": 1.8},
    "Mocha Java": {"Value": 90, "Weight": 1.2},
    "Hazelnut Flavor": {"Value": 95, "Weight": 1.0},
    "Caramel Blend": {"Value": 105, "Weight": 1.3},
    "Vanilla Flavor": {"Value": 85, "Weight": 1.2},
    "Cappuccino Mix": {"Value": 140, "Weight": 1.5},
    "Pumpkin Spice": {"Value": 75, "Weight": 1.1},
    "Decaf Roast": {"Value": 60, "Weight": 1.0},
    "Organic Roast": {"Value": 170, "Weight": 1.6},
    "Cold Brew": {"Value": 115, "Weight": 1.4},
    "Peruvian Blend": {"Value": 155, "Weight": 1.7},
    "Kenyan AA": {"Value": 125, "Weight": 1.3}
  }
}