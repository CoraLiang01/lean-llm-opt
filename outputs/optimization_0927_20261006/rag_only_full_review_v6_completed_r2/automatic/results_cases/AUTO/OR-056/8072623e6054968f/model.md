Let:
- \( I \) be the set of display areas, indexed by \( i \), with DisplayID from capacity.csv.
- \( J \) be the set of vessel types, indexed by \( j \), with ProductName from products.csv.
- \( C_i \) be the capacity of display area \( i \) (from the Capacity column in capacity.csv).
- \( v_j \) be the value of vessel type \( j \) (from the Value column in products.csv).
- \( w_j \) be the size (weight) of vessel type \( j \) (from the Weight column in products.csv).
- \( x_{ij} \) be the integer decision variable: number of vessels of type \( j \) to place in display area \( i \).

Sets and parameters (in source order):

Display areas (from capacity.csv):
\[
\begin{array}{ll}
\text{DisplayID} & \text{Capacity} \\
1 & 457 \\
2 & 604 \\
3 & 751 \\
4 & 468 \\
5 & 343 \\
6 & 408 \\
7 & 741 \\
8 & 914 \\
9 & 682 \\
10 & 409 \\
11 & 342 \\
12 & 903 \\
13 & 680 \\
14 & 886 \\
\end{array}
\]

Vessel types (from products.csv):
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Decision variables:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]
where \( x_{ij} \) is the number of vessels of type \( j \) placed in display area \( i \).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]
That is, maximize the total value of all vessels placed in all display areas.

Constraints:
For each display area \( i \) (DisplayID), the total size of vessels assigned cannot exceed its capacity:
\[
\sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i \in I
\]

Variable domains:
\[
x_{ij} \in \{0, 1, 2, \ldots\} \quad \forall i \in I,\, j \in J
\]

Full numerical model (with explicit indices):

Let \( I = \{1,2,\ldots,14\} \) (DisplayID), \( J \) as ordered above.

\[
\begin{align*}
\max\ & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{1j} \leq 457 \\
& \sum_{j=1}^{20} w_j x_{2j} \leq 604 \\
& \sum_{j=1}^{20} w_j x_{3j} \leq 751 \\
& \sum_{j=1}^{20} w_j x_{4j} \leq 468 \\
& \sum_{j=1}^{20} w_j x_{5j} \leq 343 \\
& \sum_{j=1}^{20} w_j x_{6j} \leq 408 \\
& \sum_{j=1}^{20} w_j x_{7j} \leq 741 \\
& \sum_{j=1}^{20} w_j x_{8j} \leq 914 \\
& \sum_{j=1}^{20} w_j x_{9j} \leq 682 \\
& \sum_{j=1}^{20} w_j x_{10j} \leq 409 \\
& \sum_{j=1}^{20} w_j x_{11j} \leq 342 \\
& \sum_{j=1}^{20} w_j x_{12j} \leq 903 \\
& \sum_{j=1}^{20} w_j x_{13j} \leq 680 \\
& \sum_{j=1}^{20} w_j x_{14j} \leq 886 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,14;\ j=1,\ldots,20
\end{align*}
\]
where for each \( j \), \( v_j \) and \( w_j \) are as listed above in the same order as in products.csv.

This model maximizes the total value of boats assigned to display areas, subject to each area's capacity, with integer numbers of each vessel type per area.