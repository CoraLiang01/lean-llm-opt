Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$.

Indices:
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

Parameters:
- $c_i$: Capacity of section $i$ (from capacity.csv)
- $p_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight (space requirement) of product $j$ (from products.csv)

Data:

From capacity.csv:
\[
\begin{array}{ll}
\text{SectionID} & \text{Capacity} \\
1 & 100 \\
2 & 150 \\
3 & 120 \\
4 & 130 \\
5 & 90 \\
6 & 110 \\
7 & 160 \\
8 & 140 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
1 & 10 & 2 \\
2 & 15 & 3 \\
3 & 8 & 1 \\
4 & 12 & 2 \\
5 & 20 & 4 \\
6 & 25 & 5 \\
7 & 5 & 1 \\
8 & 30 & 6 \\
9 & 18 & 3 \\
10 & 22 & 4 \\
\end{array}
\]

Model:

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} p_j \cdot x_{ij}
\]
where $p_j$ is the Value of product $j$.

Subject to:

Section capacity constraints:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,8
\]
where $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of section $i$.

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8;\; j = 1,\ldots,10
\]

Where:
- $x_{ij}$: number of units of product $j$ to be placed in section $i$ (decision variable)
- $p_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight (space requirement) of product $j$ (from products.csv)
- $c_i$: Capacity of section $i$ (from capacity.csv)