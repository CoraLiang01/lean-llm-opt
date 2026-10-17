Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Define:
- Sections $i \in \{1,2,3,4,5,6,7,8\}$ (from SectionID in capacity.csv)
- Products $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (from ProductName in products.csv)
- $v_j$ = Value of product $j$
- $w_j$ = Weight (space requirement) of product $j$
- $C_i$ = Capacity of section $i$

Parameters from the data:

Section capacities:
- $C_1 = 100$
- $C_2 = 150$
- $C_3 = 120$
- $C_4 = 130$
- $C_5 = 90$
- $C_6 = 110$
- $C_7 = 160$
- $C_8 = 140$

Product values and weights:
- $v_1 = 10$, $w_1 = 2$
- $v_2 = 15$, $w_2 = 3$
- $v_3 = 8$, $w_3 = 1$
- $v_4 = 12$, $w_4 = 2$
- $v_5 = 20$, $w_5 = 4$
- $v_6 = 25$, $w_6 = 5$
- $v_7 = 5$, $w_7 = 1$
- $v_8 = 30$, $w_8 = 6$
- $v_9 = 18$, $w_9 = 3$
- $v_{10} = 22$, $w_{10} = 4$

The complete mathematical model is:

$$
\text{Maximize} \quad \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
$$

Subject to, for each section $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

Where:

- $v_j$ and $w_j$ are as listed above for each product $j$.
- $C_i$ is as listed above for each section $i$.