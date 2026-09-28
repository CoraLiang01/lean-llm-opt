##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to be placed in section $i$, for all sections $i \in S$ and products $j \in P$.

##### Parameters

Sections $S = \{1,2,3,4,5,6,7,8\}$, with capacities:
- $C_1 = 100$
- $C_2 = 150$
- $C_3 = 120$
- $C_4 = 130$
- $C_5 = 90$
- $C_6 = 110$
- $C_7 = 160$
- $C_8 = 140$

Products $P = \{1,2,3,4,5,6,7,8,9,10\}$, with values and weights:
- Product 1: $v_1 = 10$, $w_1 = 2$
- Product 2: $v_2 = 15$, $w_2 = 3$
- Product 3: $v_3 = 8$, $w_3 = 1$
- Product 4: $v_4 = 12$, $w_4 = 2$
- Product 5: $v_5 = 20$, $w_5 = 4$
- Product 6: $v_6 = 25$, $w_6 = 5$
- Product 7: $v_7 = 5$, $w_7 = 1$
- Product 8: $v_8 = 30$, $w_8 = 6$
- Product 9: $v_9 = 18$, $w_9 = 3$
- Product 10: $v_{10} = 22$, $w_{10} = 4$

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. Section capacity constraints:
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S
   \]
2. Integer and nonnegativity constraints:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   \]

##### Full Parameter Listing

- $S = \{1,2,3,4,5,6,7,8\}$
- $C = [100, 150, 120, 130, 90, 110, 160, 140]$
- $P = \{1,2,3,4,5,6,7,8,9,10\}$
- $v = [10, 15, 8, 12, 20, 25, 5, 30, 18, 22]$
- $w = [2, 3, 1, 2, 4, 5, 1, 6, 3, 4]$

##### Mathematical Model

\[
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad i=1,\ldots,8;\ j=1,\ldots,10
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above, and $C_i$ are the section capacities.