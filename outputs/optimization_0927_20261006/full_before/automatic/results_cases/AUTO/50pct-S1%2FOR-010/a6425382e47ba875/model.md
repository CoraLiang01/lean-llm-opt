Let $x_{ij}$ be the number of units of product $j$ (ProductName $j$) to be placed in section $i$ (SectionID $i$). All $x_{ij}$ are nonnegative integers.

Parameters (from the data):

Sections (SectionID, Capacity):

- 1: 100
- 2: 150
- 3: 120
- 4: 130
- 5: 90
- 6: 110
- 7: 160
- 8: 140

Products (ProductName, Value, Weight):

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

The mathematical model is:

Objective:
$$
\max \sum_{i \in \{1,2,3,4,5,6,7,8\}} \sum_{j \in \{1,2,3,4,5,6,7,8,9,10\}} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to, for each section $i$ (SectionID):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

where $w_j$ is the Weight (space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,3,4,5,6,7,8\},\ j \in \{1,2,3,4,5,6,7,8,9,10\}
$$

Numerical parameters:

- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$
- $v_1 = 10$, $v_2 = 15$, $v_3 = 8$, $v_4 = 12$, $v_5 = 20$, $v_6 = 25$, $v_7 = 5$, $v_8 = 30$, $v_9 = 18$, $v_{10} = 22$
- $w_1 = 2$, $w_2 = 3$, $w_3 = 1$, $w_4 = 2$, $w_5 = 4$, $w_6 = 5$, $w_7 = 1$, $w_8 = 6$, $w_9 = 3$, $w_{10} = 4$

Explicitly, for each section $i$:

Section 1:
$$
2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 4x_{1,5} + 5x_{1,6} + 1x_{1,7} + 6x_{1,8} + 3x_{1,9} + 4x_{1,10} \leq 100
$$

Section 2:
$$
2x_{2,1} + 3x_{2,2} + 1x_{2,3} + 2x_{2,4} + 4x_{2,5} + 5x_{2,6} + 1x_{2,7} + 6x_{2,8} + 3x_{2,9} + 4x_{2,10} \leq 150
$$

Section 3:
$$
2x_{3,1} + 3x_{3,2} + 1x_{3,3} + 2x_{3,4} + 4x_{3,5} + 5x_{3,6} + 1x_{3,7} + 6x_{3,8} + 3x_{3,9} + 4x_{3,10} \leq 120
$$

Section 4:
$$
2x_{4,1} + 3x_{4,2} + 1x_{4,3} + 2x_{4,4} + 4x_{4,5} + 5x_{4,6} + 1x_{4,7} + 6x_{4,8} + 3x_{4,9} + 4x_{4,10} \leq 130
$$

Section 5:
$$
2x_{5,1} + 3x_{5,2} + 1x_{5,3} + 2x_{5,4} + 4x_{5,5} + 5x_{5,6} + 1x_{5,7} + 6x_{5,8} + 3x_{5,9} + 4x_{5,10} \leq 90
$$

Section 6:
$$
2x_{6,1} + 3x_{6,2} + 1x_{6,3} + 2x_{6,4} + 4x_{6,5} + 5x_{6,6} + 1x_{6,7} + 6x_{6,8} + 3x_{6,9} + 4x_{6,10} \leq 110
$$

Section 7:
$$
2x_{7,1} + 3x_{7,2} + 1x_{7,3} + 2x_{7,4} + 4x_{7,5} + 5x_{7,6} + 1x_{7,7} + 6x_{7,8} + 3x_{7,9} + 4x_{7,10} \leq 160
$$

Section 8:
$$
2x_{8,1} + 3x_{8,2} + 1x_{8,3} + 2x_{8,4} + 4x_{8,5} + 5x_{8,6} + 1x_{8,7} + 6x_{8,8} + 3x_{8,9} + 4x_{8,10} \leq 140
$$

All variables:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

Objective:
$$
\max \sum_{i=1}^8 \left(10x_{i,1} + 15x_{i,2} + 8x_{i,3} + 12x_{i,4} + 20x_{i,5} + 25x_{i,6} + 5x_{i,7} + 30x_{i,8} + 18x_{i,9} + 22x_{i,10}\right)
$$