Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Indices:
- $i$ indexes SectionID $\in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName $\in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $c_i$ = Capacity of section $i$ (from "capacity.csv")
- $v_j$ = Value (price) of product $j$ (from "products.csv")
- $w_j$ = Weight (space requirement) of product $j$ (from "products.csv")

Data:

Section Capacities:
- $c_1 = 100$
- $c_2 = 150$
- $c_3 = 120$
- $c_4 = 130$
- $c_5 = 90$
- $c_6 = 110$
- $c_7 = 160$
- $c_8 = 140$

Product Values and Weights:
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

Model:

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]

Subject to (for each section $i$):
\[
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

Where:
- $x_{ij}$: Number of units of product $j$ to stock in section $i$
- $v_j$: Value of product $j$
- $w_j$: Space requirement of product $j$
- $c_i$: Capacity of section $i$