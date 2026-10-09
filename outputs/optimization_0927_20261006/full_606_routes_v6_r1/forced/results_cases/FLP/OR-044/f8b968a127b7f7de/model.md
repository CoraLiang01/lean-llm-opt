##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to be placed in section $i$, for all sections $i \in S$ and products $j \in P$.

##### Parameters

Sections $S = \{1, 2, 3, 4, 5, 6, 7, 8\}$ with capacities:
- $C_1 = 100$
- $C_2 = 150$
- $C_3 = 120$
- $C_4 = 130$
- $C_5 = 90$
- $C_6 = 110$
- $C_7 = 160$
- $C_8 = 140$

Products $P = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$ with values and weights:
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

1. **Section Capacity Constraints** (for each section $i \in S$):

   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i
   \]

   Specifically:
   - $\sum_{j=1}^{10} w_j x_{1j} \leq 100$
   - $\sum_{j=1}^{10} w_j x_{2j} \leq 150$
   - $\sum_{j=1}^{10} w_j x_{3j} \leq 120$
   - $\sum_{j=1}^{10} w_j x_{4j} \leq 130$
   - $\sum_{j=1}^{10} w_j x_{5j} \leq 90$
   - $\sum_{j=1}^{10} w_j x_{6j} \leq 110$
   - $\sum_{j=1}^{10} w_j x_{7j} \leq 160$
   - $\sum_{j=1}^{10} w_j x_{8j} \leq 140$

2. **Integrality and Nonnegativity**:

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
   \]

##### Retrieved Information

Sections:
- 1: Capacity 100
- 2: Capacity 150
- 3: Capacity 120
- 4: Capacity 130
- 5: Capacity 90
- 6: Capacity 110
- 7: Capacity 160
- 8: Capacity 140

Products:
- 1: Value 10, Weight 2
- 2: Value 15, Weight 3
- 3: Value 8, Weight 1
- 4: Value 12, Weight 2
- 5: Value 20, Weight 4
- 6: Value 25, Weight 5
- 7: Value 5, Weight 1
- 8: Value 30, Weight 6
- 9: Value 18, Weight 3
- 10: Value 22, Weight 4