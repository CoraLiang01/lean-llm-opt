Let $x_{ij}$ be the number of units of boat type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

Sets:
- $i \in \{\text{1}, \text{2}, \ldots, \text{14}\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

Parameters:
- $c_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of boat type $j$ (from products.csv)
- $w_j$ = Weight (size) of boat type $j$ (from products.csv)

Objective:
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to, for each display area $i$ (DisplayID):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,14\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,14\},\; \forall j \in \{1,2,\ldots,20\}
\]

Where the data is:

Display areas and capacities (from capacity.csv, in source order):

\[
\begin{array}{ll}
\text{DisplayID} & \text{Capacity} \\
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

Boat types, values, and weights (from products.csv, in source order):

\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Summary of the complete model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,14;\ j=1,\ldots,20
\end{align*}
\]

with all identifiers and coefficients as listed above.