Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let:
- $S$ = set of SectionIDs: $\{1,2,3,4,5,6,7,8\}$
- $P$ = set of ProductNames: $\{1,2,3,4,5,6,7,8,9,10\}$
- $c_i$ = Capacity of section $i$ (from "capacity.csv")
- $v_j$ = Value (price) of product $j$ (from "products.csv")
- $w_j$ = Weight (space requirement) of product $j$ (from "products.csv")

Parameters (from the data):

Section Capacities:
\[
\begin{align*}
c_1 &= 100 \\
c_2 &= 150 \\
c_3 &= 120 \\
c_4 &= 130 \\
c_5 &= 90 \\
c_6 &= 110 \\
c_7 &= 160 \\
c_8 &= 140 \\
\end{align*}
\]

Product Values and Weights:
\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
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

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to:

Section Capacity Constraints (for each section $i$):
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
\]

Integrality and Nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

Where:
- $x_{ij}$: number of units of product $j$ to be placed in section $i$
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight (space requirement) of product $j$ (see table above)
- $c_i$: capacity of section $i$ (see list above)

All identifiers and coefficients are as retrieved and preserved in source order.