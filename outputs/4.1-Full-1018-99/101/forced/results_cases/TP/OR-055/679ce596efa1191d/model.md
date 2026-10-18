##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ be the set of display areas (DisplayID in source order).

Let $J =$ 
{
Speedboat,
Fishing Boat,
Catamaran,
Yacht,
Sailboat,
Kayak,
Canoe,
Houseboat,
Pontoon,
Jet Ski,
Rowboat,
Hovercraft,
Cabin Cruiser,
Wakeboard Boat,
Dinghy,
Trawler,
Paddle Boat,
Submarine,
RIB,
Skiff
}
(ProductName in source order).

##### Parameters

For each display area $i \in I$:

- $C_i$ = capacity of display area $i$:

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

For each boat type $j \in J$:

- $v_j$ = value of one unit of boat type $j$
- $w_j$ = size (weight) of one unit of boat type $j$

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

For each $i \in I$, $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of boat type $j$ placed in display area $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each display area $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Complete Model

\[
\begin{align*}
\max\ & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,14,\ j=1,\ldots,20
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above, and $C_i$ as listed above, with $x_{ij}$ integer and nonnegative.