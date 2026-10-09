##### Sets

Let $I = \{1,2,\ldots,14\}$ be the set of display areas (DisplayID from 1 to 14, in source order).

Let $J =$ 
{Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff}
(listed in source order from products.csv, 21 products).

##### Parameters

For each display area $i \in I$:

- $C_i$ = capacity of display area $i$.

From capacity.csv (source order):

\[
\begin{align*}
C_1 &= 356 \\
C_2 &= 478 \\
C_3 &= 305 \\
C_4 &= 291 \\
C_5 &= 168 \\
C_6 &= 449 \\
C_7 &= 139 \\
C_8 &= 383 \\
C_9 &= 472 \\
C_{10} &= 288 \\
C_{11} &= 320 \\
C_{12} &= 250 \\
C_{13} &= 402 \\
C_{14} &= 293 \\
\end{align*}
\]

For each product $j \in J$:

- $v_j$ = value of one unit of product $j$
- $w_j$ = size (weight) of one unit of product $j$

From products.csv (source order):

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\hline
\text{Speedboat} & 69978 & 18 \\
\text{Fishing Boat} & 54011 & 42 \\
\text{Catamaran} & 36352 & 49 \\
\text{Yacht} & 51521 & 42 \\
\text{Sailboat} & 50415 & 41 \\
\text{Kayak} & 76109 & 48 \\
\text{Canoe} & 50462 & 22 \\
\text{Houseboat} & 28989 & 29 \\
\text{Pontoon} & 23318 & 45 \\
\text{Jet Ski} & 26142 & 14 \\
\text{Rowboat} & 42040 & 38 \\
\text{Hovercraft} & 85961 & 47 \\
\text{Cabin Cruiser} & 50142 & 45 \\
\text{Wakeboard Boat} & 48478 & 28 \\
\text{Dinghy} & 60953 & 24 \\
\text{Trawler} & 95265 & 39 \\
\text{Paddle Boat} & 22839 & 32 \\
\text{Submarine} & 90957 & 36 \\
\text{RIB} & 84652 & 14 \\
\text{Skiff} & 78991 & 16 \\
\end{array}
\]

##### Decision Variables

For each display area $i \in I$ and product $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in display area $i$

##### Objective

\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]

##### Constraints

For each display area $i \in I$:

\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Complete Numerical Formulation

Sets:

- $I = \{1,2,\ldots,14\}$
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

Parameters:

- $C_1 = 356$, $C_2 = 478$, $C_3 = 305$, $C_4 = 291$, $C_5 = 168$, $C_6 = 449$, $C_7 = 139$, $C_8 = 383$, $C_9 = 472$, $C_{10} = 288$, $C_{11} = 320$, $C_{12} = 250$, $C_{13} = 402$, $C_{14} = 293$
- $(v_j, w_j)$ for $j \in J$ as listed above

Variables:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i \in I$, $j \in J$

Objective:

\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to:

\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,14
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,14,\ j = 1,\ldots,20
\]

###### Retrieved Information

capacity.csv (source order):

1. DisplayID: 1, Capacity: 356
2. DisplayID: 2, Capacity: 478
3. DisplayID: 3, Capacity: 305
4. DisplayID: 4, Capacity: 291
5. DisplayID: 5, Capacity: 168
6. DisplayID: 6, Capacity: 449
7. DisplayID: 7, Capacity: 139
8. DisplayID: 8, Capacity: 383
9. DisplayID: 9, Capacity: 472
10. DisplayID: 10, Capacity: 288
11. DisplayID: 11, Capacity: 320
12. DisplayID: 12, Capacity: 250
13. DisplayID: 13, Capacity: 402
14. DisplayID: 14, Capacity: 293

products.csv (source order):

1. Speedboat, Value: 69978, Weight: 18
2. Fishing Boat, Value: 54011, Weight: 42
3. Catamaran, Value: 36352, Weight: 49
4. Yacht, Value: 51521, Weight: 42
5. Sailboat, Value: 50415, Weight: 41
6. Kayak, Value: 76109, Weight: 48
7. Canoe, Value: 50462, Weight: 22
8. Houseboat, Value: 28989, Weight: 29
9. Pontoon, Value: 23318, Weight: 45
10. Jet Ski, Value: 26142, Weight: 14
11. Rowboat, Value: 42040, Weight: 38
12. Hovercraft, Value: 85961, Weight: 47
13. Cabin Cruiser, Value: 50142, Weight: 45
14. Wakeboard Boat, Value: 48478, Weight: 28
15. Dinghy, Value: 60953, Weight: 24
16. Trawler, Value: 95265, Weight: 39
17. Paddle Boat, Value: 22839, Weight: 32
18. Submarine, Value: 90957, Weight: 36
19. RIB, Value: 84652, Weight: 14
20. Skiff, Value: 78991, Weight: 16