##### Decision Variables

Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ denote the number of vessels of type $j$ (product $j$) to be placed in display area $i$, for all display areas $i$ and vessel types $j$.

##### Parameters

Let $I = \{1,2,\ldots,14\}$ be the set of display areas (from capacity.csv).  
Let $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (from products.csv).

Let $C_i$ be the capacity of display area $i$:

\[
\begin{align*}
C_1 &= 457 \\
C_2 &= 604 \\
C_3 &= 751 \\
C_4 &= 468 \\
C_5 &= 343 \\
C_6 &= 408 \\
C_7 &= 741 \\
C_8 &= 914 \\
C_9 &= 682 \\
C_{10} &= 409 \\
C_{11} &= 342 \\
C_{12} &= 903 \\
C_{13} &= 680 \\
C_{14} &= 886 \\
\end{align*}
\]

Let $v_j$ be the value and $w_j$ the weight (size) of vessel type $j$ (in source order):

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\hline
\text{Speedboat} & 29664 & 18 \\
\text{Fishing Boat} & 31778 & 36 \\
\text{Catamaran} & 73501 & 25 \\
\text{Yacht} & 78255 & 16 \\
\text{Sailboat} & 93606 & 97 \\
\text{Kayak} & 46983 & 35 \\
\text{Canoe} & 95026 & 32 \\
\text{Houseboat} & 57685 & 100 \\
\text{Pontoon} & 60323 & 43 \\
\text{Jet Ski} & 91224 & 15 \\
\text{Rowboat} & 44003 & 95 \\
\text{Hovercraft} & 75998 & 57 \\
\text{Cabin Cruiser} & 84525 & 13 \\
\text{Wakeboard Boat} & 66207 & 44 \\
\text{Dinghy} & 65002 & 64 \\
\text{Trawler} & 33132 & 88 \\
\text{Paddle Boat} & 69239 & 42 \\
\text{Submarine} & 66948 & 46 \\
\text{RIB} & 88240 & 24 \\
\text{Skiff} & 48858 & 93 \\
\end{array}
\]

##### Objective Function

\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]

##### Constraints

For each display area $i=1,\ldots,14$:

\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
\]

For all $i=1,\ldots,14$, $j=1,\ldots,20$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,14,\ j=1,\ldots,20
\end{align*}
\]

Where:

- $C_i$ as above for $i=1,\ldots,14$
- $v_j$, $w_j$ as above for $j=1,\ldots,20$ (in source order)
- $x_{ij}$: number of vessels of type $j$ in display area $i$ (integer, nonnegative)