Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Sets:
- $i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ (StorageID from capacity.csv)
- $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ (ProductName from products.csv)

Parameters:
- $c_i$ = Capacity of storage area $i$ (from capacity.csv)
- $v_j$ = Value of air conditioner type $j$ (from products.csv)
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv)

Data:

Storage Areas and Capacities:
\[
\begin{array}{ll}
\text{StorageID} & \text{Capacity} \\
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

Air Conditioner Types, Values, and Weights:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Model:

Objective:
\[
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

Subject to:

Capacity constraints for each storage area:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $v_j$ and $w_j$ are as listed above for each ProductName $j$
- $c_i$ is as listed above for each StorageID $i$