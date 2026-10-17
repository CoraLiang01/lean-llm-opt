Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and nonnegative.

Define:
- $S$ = set of storage areas (StorageID from capacity.csv): $\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $P$ = set of air conditioner types (ProductName from products.csv): 
  - Window Unit
  - Portable Unit
  - Split System
  - Ductless System
  - Central AC
  - Hybrid AC
  - Geothermal AC
  - Smart AC
  - Evaporative Cooler
  - Package Unit

Let:
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of storage area $i$ (from capacity.csv)

The model is:

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

Where the parameters are:

Storage areas and capacities:
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

Product types, values, and weights:
\[
\begin{array}{lll}
\text{ProductName} & v_j\ (\text{Value}) & w_j\ (\text{Weight}) \\
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

Decision variables:
\[
x_{ij} = \text{number of units of product } j \text{ in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Full model:
\[
\begin{align*}
\max\ & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\ & \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,15,\ j=1,\ldots,10
\end{align*}
\]
with all $v_j$, $w_j$, $C_i$ as listed above.