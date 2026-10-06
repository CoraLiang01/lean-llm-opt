Let:
- \( x_{ij} \): number of units of product \( j \) placed in cabinet \( i \), for \( i \in \{1,2,\ldots,10\} \) and \( j \in \{1,2,\ldots,18\} \) (corresponding to the order of cabinets and products as given).

Indices:
- Cabinets: \( i \) (CabinetID from 1 to 10, as in capacity.csv)
- Products: \( j \) (ProductName from products.csv, in the given order)

Parameters:
- \( C_i \): Capacity of cabinet \( i \) (from capacity.csv)
- \( v_j \): Value per unit of product \( j \) (from products.csv)
- \( w_j \): Weight per unit of product \( j \) (from products.csv)

Data:
From capacity.csv:
\[
\begin{array}{ll}
\text{CabinetID} & C_i \\
1 & 400 \\
2 & 600 \\
3 & 500 \\
4 & 700 \\
5 & 450 \\
6 & 650 \\
7 & 550 \\
8 & 750 \\
9 & 480 \\
10 & 520 \\
\end{array}
\]

From products.csv (in order):
\[
\begin{array}{lll}
j & \text{ProductName} & (v_j, w_j) \\
1 & \text{Espresso Beans} & (100, 1.0) \\
2 & \text{Colombian Roast} & (150, 1.5) \\
3 & \text{Arabica Blend} & (80, 1.2) \\
4 & \text{French Roast} & (120, 1.3) \\
5 & \text{Italian Roast} & (130, 1.4) \\
6 & \text{House Blend} & (110, 1.1) \\
7 & \text{Sumatra Coffee} & (160, 1.8) \\
8 & \text{Mocha Java} & (90, 1.2) \\
9 & \text{Hazelnut Flavor} & (95, 1.0) \\
10 & \text{Caramel Blend} & (105, 1.3) \\
11 & \text{Vanilla Flavor} & (85, 1.2) \\
12 & \text{Cappuccino Mix} & (140, 1.5) \\
13 & \text{Pumpkin Spice} & (75, 1.1) \\
14 & \text{Decaf Roast} & (60, 1.0) \\
15 & \text{Organic Roast} & (170, 1.6) \\
16 & \text{Cold Brew} & (115, 1.4) \\
17 & \text{Peruvian Blend} & (155, 1.7) \\
18 & \text{Kenyan AA} & (125, 1.3) \\
\end{array}
\]

Model:

Decision variables:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\]

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij}
\]
where \( v_j \) is as above.

Subject to (for each cabinet \( i \)):
\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i \quad \forall i \in \{1,\ldots,10\}
\]
where \( w_j \) and \( C_i \) are as above.

Variable domains:
\[
x_{ij} \in \{0,1,2,\ldots\} \quad \forall i,j
\]

Explicitly, for each cabinet \( i \) (CabinetID as in capacity.csv), the constraint is:
\[
\sum_{j=1}^{18} w_j x_{ij} \leq C_i
\]
with the weights \( w_j \) and values \( v_j \) as listed above for each product \( j \).

This is a multiple knapsack integer programming problem, maximizing total value placed in all cabinets, subject to each cabinet's weight capacity, with integer numbers of units for each product in each cabinet.