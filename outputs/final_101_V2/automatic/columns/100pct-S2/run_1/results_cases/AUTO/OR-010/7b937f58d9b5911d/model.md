Let $x_{ij}$ = number of units of product $j$ to be placed in section $i$.

Indices:
- $i$ indexes SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from products.csv)
- $C_i$ = Capacity of section $i$ (from capacity.csv)

Data:

Section Capacities:
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

Product Values and Space Requirements:
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
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]
where $v_j$ is as above.

Subject to, for each section $i$:
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]
where $w_j$ and $C_i$ are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

All parameters and indices are as retrieved and used in the model.