Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv) and $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv).

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$:

\[
(v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_{10}) = (10, 15, 8, 12, 20, 25, 5, 30, 18, 22)
\]

Subject to:

For each section $i$ (SectionID from 1 to 8), the total space used does not exceed its Capacity:

\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]
where $w_j$ is the Weight (space requirement) of product $j$ and $c_i$ is the Capacity of section $i$:

\[
\begin{align*}
&c_1 = 100 \\
&c_2 = 150 \\
&c_3 = 120 \\
&c_4 = 130 \\
&c_5 = 90 \\
&c_6 = 110 \\
&c_7 = 160 \\
&c_8 = 140 \\
\end{align*}
\]

\[
(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_{10}) = (2, 3, 1, 2, 4, 5, 1, 6, 3, 4)
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

All variables, coefficients, and constraints are as retrieved and indexed by the original SectionID and ProductName.