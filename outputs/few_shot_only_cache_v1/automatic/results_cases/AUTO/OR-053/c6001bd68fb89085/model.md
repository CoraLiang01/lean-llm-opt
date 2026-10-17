Let $x_{ij}$ be the integer number of units of product $j$ (ProductName $j$ from products.csv) to be placed on shelf $i$ (ShelfID $i$ from capacity.csv).

Parameters:
- $V_j$: Value of product $j$ (from products.csv, column "Value")
- $W_j$: Weight of product $j$ (from products.csv, column "Weight")
- $C_i$: Capacity of shelf $i$ (from capacity.csv, column "Capacity")

Indices:
- $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$ (ShelfID from capacity.csv, in source order)
- $j \in \{\text{1}, \text{2}, \ldots, \text{20}\}$ (ProductName from products.csv, in source order)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} V_j \cdot x_{ij}
\]

Subject to, for each shelf $i$:
\[
\sum_{j=1}^{20} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,20\}
\]

Where the parameters are as follows (in source order):

Shelf capacities ($C_i$):

\[
\begin{array}{ll}
C_1 = 500 & (\text{ShelfID } 1) \\
C_2 = 700 & (\text{ShelfID } 2) \\
C_3 = 600 & (\text{ShelfID } 3) \\
C_4 = 800 & (\text{ShelfID } 4) \\
C_5 = 550 & (\text{ShelfID } 5) \\
C_6 = 900 & (\text{ShelfID } 6) \\
C_7 = 650 & (\text{ShelfID } 7) \\
C_8 = 750 & (\text{ShelfID } 8) \\
C_9 = 820 & (\text{ShelfID } 9) \\
C_{10} = 570 & (\text{ShelfID } 10) \\
\end{array}
\]

Product values ($V_j$) and weights ($W_j$):

\[
\begin{array}{lll}
j & V_j & W_j \\
1 & 50 & 10 \\
2 & 70 & 20 \\
3 & 30 & 5 \\
4 & 60 & 15 \\
5 & 80 & 25 \\
6 & 90 & 30 \\
7 & 40 & 12 \\
8 & 100 & 35 \\
9 & 55 & 10 \\
10 & 75 & 20 \\
11 & 65 & 18 \\
12 & 95 & 28 \\
13 & 45 & 8 \\
14 & 85 & 22 \\
15 & 70 & 25 \\
16 & 110 & 40 \\
17 & 50 & 14 \\
18 & 60 & 16 \\
19 & 120 & 50 \\
20 & 100 & 30 \\
\end{array}
\]

Summary:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} V_j x_{ij} \\
\text{s.t.}\ & \sum_{j=1}^{20} W_j x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
\]