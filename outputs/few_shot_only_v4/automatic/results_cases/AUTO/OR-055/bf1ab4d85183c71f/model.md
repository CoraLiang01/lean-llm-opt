Let $x_{ij}$ be the number of units of boat type $j$ to be placed in display area $i$. All $x_{ij}$ are nonnegative integers.

Let $I$ be the set of display areas (DisplayID from capacity.csv), and $J$ be the set of boat types (ProductName from products.csv).

Let $c_i$ be the capacity of display area $i$ (from capacity.csv).
Let $w_j$ be the weight (size) of boat type $j$ (from products.csv).
Let $v_j$ be the value of boat type $j$ (from products.csv).

Sets:
- $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

Parameters (from products.csv):

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

Parameters (from capacity.csv):

\[
\begin{array}{ll}
\text{DisplayID} & c_i \\
\hline
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

Model:

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]