Let $x_{ij}$ be the number of units of air conditioner type $j$ placed in storage area $i$.

Indices:
- $i$ indexes StorageID: $i \in \{1,2,\ldots,15\}$
- $j$ indexes ProductName (air conditioner types): $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$

Parameters:
- $v_j$: Value of one unit of product $j$
- $w_j$: Weight (size) of one unit of product $j$
- $C_i$: Capacity of storage area $i$

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

Subject to, for each storage area $i$:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $v_j$ and $w_j$ are as given in the table above for each ProductName $j$.
- $C_i$ is as given in the table above for each StorageID $i$.

All variables, parameters, and constraints use the exact identifiers and coefficients from the provided data.