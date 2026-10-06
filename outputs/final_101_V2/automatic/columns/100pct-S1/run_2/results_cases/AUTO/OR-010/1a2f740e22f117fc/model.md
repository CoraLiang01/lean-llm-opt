Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

Parameters:
- Let $S$ be the set of SectionIDs: $\{1,2,3,4,5,6,7,8\}$
- Let $P$ be the set of ProductNames: $\{1,2,3,4,5,6,7,8,9,10\}$
- Let $c_i$ be the Capacity of section $i$ (from "capacity.csv")
- Let $v_j$ be the Value of product $j$ (from "products.csv")
- Let $w_j$ be the Weight (shelf space requirement) of product $j$ (from "products.csv")

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

Product Values and Weights:
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
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each section $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

Where:
- $x_{ij}$: Number of units of product $j$ to be placed in section $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight (shelf space requirement) of product $j$ (see table above)
- $c_i$: Capacity of section $i$ (see table above)