Let:
- I = set of storage areas, indexed by i (from StorageID in capacity.csv)
- J = set of air conditioner types, indexed by j (from ProductName in products.csv)
- C_i = capacity of storage area i (from Capacity in capacity.csv)
- v_j = value of air conditioner type j (from Value in products.csv)
- w_j = size (weight) of air conditioner type j (from Weight in products.csv)
- x_{ij} = number of units of air conditioner type j to be placed in storage area i (decision variable, integer, x_{ij} ≥ 0)

Indices:
- i ∈ {1, 2, ..., 15} (StorageID from capacity.csv)
- j ∈ {1, 2, ..., 10} (ProductName from products.csv, see mapping below)

Mapping for j:
1: Window Unit
2: Portable Unit
3: Split System
4: Ductless System
5: Central AC
6: Hybrid AC
7: Geothermal AC
8: Smart AC
9: Evaporative Cooler
10: Package Unit

Parameters:
From capacity.csv:
StorageID | Capacity
1 | 1083
2 | 1840
3 | 770
4 | 1299
5 | 1259
6 | 543
7 | 1831
8 | 855
9 | 619
10 | 637
11 | 935
12 | 626
13 | 1457
14 | 1198
15 | 837

From products.csv:
ProductName | Value | Weight
Window Unit | 4811 | 114
Portable Unit | 1130 | 200
Split System | 1611 | 106
Ductless System | 3368 | 256
Central AC | 2135 | 268
Hybrid AC | 1046 | 185
Geothermal AC | 4030 | 299
Smart AC | 3761 | 131
Evaporative Cooler | 3523 | 139
Package Unit | 1701 | 105

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,15}, j ∈ {1,...,10}

Objective:
Maximize total value of air conditioners allocated:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
\]
where v_j is as above.

Subject to:

For each storage area i ∈ {1,...,15}:
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i
\]
where w_j is as above and C_i is the capacity for storage area i.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,15\},\ j \in \{1,...,10\}
\]

Explicitly, the model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij} \\
\text{subject to} \quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \quad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,15;\ j = 1,\ldots,10
\end{align*}
\]

Where:
- v_j = [4811, 1130, 1611, 3368, 2135, 1046, 4030, 3761, 3523, 1701] for j = 1 to 10 (in the order above)
- w_j = [114, 200, 106, 256, 268, 185, 299, 131, 139, 105] for j = 1 to 10 (in the order above)
- C_i as listed above for i = 1 to 15 (by StorageID)

This is a multiple-choice multi-knapsack integer programming model, maximizing total value subject to storage area capacities, with integer decision variables x_{ij}.