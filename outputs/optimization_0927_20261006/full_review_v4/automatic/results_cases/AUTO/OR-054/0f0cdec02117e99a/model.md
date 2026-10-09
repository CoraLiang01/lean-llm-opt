Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$.

Indices:
- $i$ indexes shelves, with ShelfID from 1 to 10.
- $j$ indexes products, with ProductName from 1 to 20.

Parameters (from the retrieved data):

Shelf capacities:
\[
\begin{align*}
&\text{Shelf 1: } C_1 = 750 \\
&\text{Shelf 2: } C_2 = 820 \\
&\text{Shelf 3: } C_3 = 570 \\
&\text{Shelf 4: } C_4 = 800 \\
&\text{Shelf 5: } C_5 = 550 \\
&\text{Shelf 6: } C_6 = 900 \\
&\text{Shelf 7: } C_7 = 650 \\
&\text{Shelf 8: } C_8 = 800 \\
&\text{Shelf 9: } C_9 = 850 \\
&\text{Shelf 10: } C_{10} = 900 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value } (v_j) & \text{Weight } (w_j) \\
1 & 55 & 10 \\
2 & 75 & 20 \\
3 & 65 & 5 \\
4 & 60 & 15 \\
5 & 80 & 25 \\
6 & 90 & 35 \\
7 & 40 & 45 \\
8 & 100 & 55 \\
9 & 55 & 65 \\
10 & 75 & 20 \\
11 & 110 & 18 \\
12 & 50 & 28 \\
13 & 60 & 8 \\
14 & 120 & 28 \\
15 & 70 & 25 \\
16 & 110 & 40 \\
17 & 50 & 55 \\
18 & 60 & 70 \\
19 & 120 & 85 \\
20 & 100 & 100 \\
\end{array}
\]

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Where:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ (nonnegative integer)
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight of product $j$ (see table above)
- $C_i$: capacity of shelf $i$ (see list above)

All data and identifiers are preserved in original order as retrieved.