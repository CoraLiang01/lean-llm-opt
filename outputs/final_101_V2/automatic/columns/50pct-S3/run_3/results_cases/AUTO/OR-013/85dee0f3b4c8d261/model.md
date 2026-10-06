Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$:

\[
\begin{align*}
\text{Window Unit} &: 4811 \\
\text{Portable Unit} &: 1130 \\
\text{Split System} &: 1611 \\
\text{Ductless System} &: 3368 \\
\text{Central AC} &: 2135 \\
\text{Hybrid AC} &: 1046 \\
\text{Geothermal AC} &: 4030 \\
\text{Smart AC} &: 3761 \\
\text{Evaporative Cooler} &: 3523 \\
\text{Package Unit} &: 1701 \\
\end{align*}
\]

Subject to, for each storage area $i$ (StorageID):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
\]

where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of storage area $i$:

\[
\begin{align*}
\text{StorageID} &\quad C_i \\
1 &\quad 1083 \\
2 &\quad 1840 \\
3 &\quad 770 \\
4 &\quad 1299 \\
5 &\quad 1259 \\
6 &\quad 543 \\
7 &\quad 1831 \\
8 &\quad 855 \\
9 &\quad 619 \\
10 &\quad 637 \\
11 &\quad 935 \\
12 &\quad 626 \\
13 &\quad 1457 \\
14 &\quad 1198 \\
15 &\quad 837 \\
\end{align*}
\]

and

\[
\begin{align*}
\text{Window Unit} &: 114 \\
\text{Portable Unit} &: 200 \\
\text{Split System} &: 106 \\
\text{Ductless System} &: 256 \\
\text{Central AC} &: 268 \\
\text{Hybrid AC} &: 185 \\
\text{Geothermal AC} &: 299 \\
\text{Smart AC} &: 131 \\
\text{Evaporative Cooler} &: 139 \\
\text{Package Unit} &: 105 \\
\end{align*}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

All indices, coefficients, and identifiers are as retrieved and in original order.