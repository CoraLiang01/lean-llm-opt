Let:
- \( I \) = set of display areas, indexed by \( i \) (DisplayID from capacity.csv)
- \( J \) = set of boat types, indexed by \( j \) (row order from products.csv)
- \( C_i \) = capacity of display area \( i \) (from capacity.csv)
- \( v_j \) = value of boat type \( j \) (from products.csv)
- \( w_j \) = weight (size) of boat type \( j \) (from products.csv)
- \( x_{ij} \) = integer number of units of boat type \( j \) placed in display area \( i \), \( x_{ij} \geq 0 \)

Indices:
- \( i \in \{1,2,\ldots,14\} \) (DisplayID from capacity.csv, in order)
- \( j \in \{1,2,\ldots,20\} \) (row order from products.csv: 1=Speedboat, 2=Fishing Boat, ..., 20=Skiff)

Parameters:
From capacity.csv:
\[
\begin{array}{ll}
\text{DisplayID} & C_i \\
1 & 356 \\
2 & 478 \\
3 & 305 \\
4 & 291 \\
5 & 168 \\
6 & 449 \\
7 & 139 \\
8 & 383 \\
9 & 472 \\
10 & 288 \\
11 & 320 \\
12 & 250 \\
13 & 402 \\
14 & 293 \\
\end{array}
\]

From products.csv (row order preserved):
\[
\begin{array}{lll}
j & v_j & w_j \\
1 & 69978 & 18 \\
2 & 54011 & 42 \\
3 & 36352 & 49 \\
4 & 51521 & 42 \\
5 & 50415 & 41 \\
6 & 76109 & 48 \\
7 & 50462 & 22 \\
8 & 28989 & 29 \\
9 & 23318 & 45 \\
10 & 26142 & 14 \\
11 & 42040 & 38 \\
12 & 85961 & 47 \\
13 & 50142 & 45 \\
14 & 48478 & 28 \\
15 & 60953 & 24 \\
16 & 95265 & 39 \\
17 & 22839 & 32 \\
18 & 90957 & 36 \\
19 & 84652 & 14 \\
20 & 78991 & 16 \\
\end{array}
\]

Decision variables:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
\]

Objective:
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to (for each display area \( i \)):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i \in \{1,\ldots,14\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
\]

Where:
- \( v_j \) and \( w_j \) are as listed above for each boat type \( j \) (in the order of products.csv).
- \( C_i \) is the capacity for each display area \( i \) (DisplayID from capacity.csv, in order).

This is a pure integer programming model maximizing the total value of boats displayed, subject to area-specific capacity constraints.