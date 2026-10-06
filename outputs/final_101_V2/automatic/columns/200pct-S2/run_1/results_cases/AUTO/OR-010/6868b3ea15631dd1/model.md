Let $x_{ij}$ = number of units of product $j$ to be placed in section $i$.

Indices:
- $i$ indexes SectionID from "capacity.csv": $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName from "products.csv": $j \in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $c_i$ = Capacity of section $i$ (from "capacity.csv")
- $v_j$ = Value (price) of product $j$ (from "products.csv")
- $a_j$ = Weight (shelf space requirement) of product $j$ (from "products.csv")

Data:

Section Capacities (from "capacity.csv"):
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

Product Values and Space Requirements (from "products.csv"):
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

Subject to (for each section $i$):

Section capacity constraints:
\[
\sum_{j=1}^{10} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,3,4,5,6,7,8\},\; j \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Where:
- $v_j$ and $a_j$ are as given above for each product $j$
- $c_i$ is as given above for each section $i$