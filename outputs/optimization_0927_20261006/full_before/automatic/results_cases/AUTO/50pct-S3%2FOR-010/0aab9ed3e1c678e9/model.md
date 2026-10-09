Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv) and $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv).

Parameters:
- $v_j$: Value (price) of product $j$ (from products.csv)
- $w_j$: Weight (space requirement) of product $j$ (from products.csv)
- $C_i$: Capacity of section $i$ (from capacity.csv)

Data:

Section Capacities (from capacity.csv):

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

Product Values and Weights (from products.csv):

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
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

Subject to:

Section capacity constraints:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,3,4,5,6,7,8\},\; j \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Where:
- $v_j$ and $w_j$ are as given above for each product $j$.
- $C_i$ is as given above for each section $i$.

All variables, parameters, and constraints use the original identifiers and coefficients as retrieved.