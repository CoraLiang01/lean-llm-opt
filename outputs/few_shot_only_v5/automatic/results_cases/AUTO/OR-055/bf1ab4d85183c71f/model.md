Let $x_{ij}$ be the number of units of boat type $j$ to be placed in display area $i$. All $x_{ij}$ are integer and nonnegative.

Indices:
- $i$ indexes DisplayID $\in \{1,2,\ldots,14\}$
- $j$ indexes ProductName (see list below)

Parameters:
- $c_i$ = Capacity of display area $i$
- $v_j$ = Value of boat type $j$
- $w_j$ = Weight (size) of boat type $j$

Data:

Display Areas (from capacity.csv):

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

Boat Types (from products.csv):

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

Model:

Objective:
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to (for each display area $i$):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,14\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
\]

Where:
- $v_j$ and $w_j$ are as given in the table above for each ProductName $j$.
- $c_i$ is the Capacity for each DisplayID $i$ as given above.