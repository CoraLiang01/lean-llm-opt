Sets and Indices:
- Let I be the set of storage areas, indexed by i, with StorageID from capacity.csv: {1, 2, ..., 15}.
- Let J be the set of air conditioner types, indexed by j, with ProductName from products.csv: {Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit}.

Parameters:
- Capacity_i: Capacity of storage area i (from capacity.csv).
- Value_j: Value of air conditioner type j (from products.csv).
- Weight_j: Size (weight) of air conditioner type j (from products.csv).

Decision Variables:
- x_ij: Number of units of air conditioner type j to place in storage area i. (x_ij ≥ 0, integer)

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{ij}
\]

Subject to storage area capacities:
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in I
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Where:

From capacity.csv:
\[
\begin{array}{ll}
\text{StorageID} & \text{Capacity}_i \\
1 & 1083 \\
2 & 1840 \\
3 & 770 \\
4 & 1299 \\
5 & 1259 \\
6 & 543 \\
7 & 1831 \\
8 & 855 \\
9 & 619 \\
10 & 637 \\
11 & 935 \\
12 & 626 \\
13 & 1457 \\
14 & 1198 \\
15 & 837 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value}_j & \text{Weight}_j \\
\text{Window Unit} & 4811 & 114 \\
\text{Portable Unit} & 1130 & 200 \\
\text{Split System} & 1611 & 106 \\
\text{Ductless System} & 3368 & 256 \\
\text{Central AC} & 2135 & 268 \\
\text{Hybrid AC} & 1046 & 185 \\
\text{Geothermal AC} & 4030 & 299 \\
\text{Smart AC} & 3761 & 131 \\
\text{Evaporative Cooler} & 3523 & 139 \\
\text{Package Unit} & 1701 & 105 \\
\end{array}
\]

Explicitly, for each storage area i (StorageID 1 to 15):
\[
114\,x_{i,\text{Window Unit}} + 200\,x_{i,\text{Portable Unit}} + 106\,x_{i,\text{Split System}} + 256\,x_{i,\text{Ductless System}} + 268\,x_{i,\text{Central AC}} + 185\,x_{i,\text{Hybrid AC}} + 299\,x_{i,\text{Geothermal AC}} + 131\,x_{i,\text{Smart AC}} + 139\,x_{i,\text{Evaporative Cooler}} + 105\,x_{i,\text{Package Unit}} \leq \text{Capacity}_i
\]
for each i = 1, ..., 15 (with the corresponding Capacity_i from above).

All x_{ij} are nonnegative integers.

Objective:
\[
\text{Maximize} \quad \sum_{i=1}^{15} \Big( 4811\,x_{i,\text{Window Unit}} + 1130\,x_{i,\text{Portable Unit}} + 1611\,x_{i,\text{Split System}} + 3368\,x_{i,\text{Ductless System}} + 2135\,x_{i,\text{Central AC}} + 1046\,x_{i,\text{Hybrid AC}} + 4030\,x_{i,\text{Geothermal AC}} + 3761\,x_{i,\text{Smart AC}} + 3523\,x_{i,\text{Evaporative Cooler}} + 1701\,x_{i,\text{Package Unit}} \Big)
\]

Subject to the above constraints and variable domains.