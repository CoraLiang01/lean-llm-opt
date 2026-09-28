##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of cabinets (with capacities as below).

Let $J =$ 
{
  "Espresso Beans",
  "Colombian Roast",
  "Arabica Blend",
  "French Roast",
  "Italian Roast",
  "House Blend",
  "Sumatra Coffee",
  "Mocha Java",
  "Hazelnut Flavor",
  "Caramel Blend",
  "Vanilla Flavor",
  "Cappuccino Mix",
  "Pumpkin Spice",
  "Decaf Roast",
  "Organic Roast",
  "Cold Brew",
  "Peruvian Blend",
  "Kenyan AA"
}
be the set of coffee products.

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

##### Objective

\[
\max \sum_{i=1}^{10} \sum_{j \in J} v_j x_{ij}
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

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big( 
100\,x_{i,\text{Espresso Beans}} + 
150\,x_{i,\text{Colombian Roast}} + 
80\,x_{i,\text{Arabica Blend}} + 
120\,x_{i,\text{French Roast}} + 
130\,x_{i,\text{Italian Roast}} + \\
&\qquad 110\,x_{i,\text{House Blend}} + 
160\,x_{i,\text{Sumatra Coffee}} + 
90\,x_{i,\text{Mocha Java}} + 
95\,x_{i,\text{Hazelnut Flavor}} + 
105\,x_{i,\text{Caramel Blend}} + \\
&\qquad 85\,x_{i,\text{Vanilla Flavor}} + 
140\,x_{i,\text{Cappuccino Mix}} + 
75\,x_{i,\text{Pumpkin Spice}} + 
60\,x_{i,\text{Decaf Roast}} + 
170\,x_{i,\text{Organic Roast}} + \\
&\qquad 115\,x_{i,\text{Cold Brew}} + 
155\,x_{i,\text{Peruvian Blend}} + 
125\,x_{i,\text{Kenyan AA}}
\Big) \\[2ex]
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10,\ j \in J
\end{align*}
\]

Where $C_i$ and $w_j$ are as listed above.