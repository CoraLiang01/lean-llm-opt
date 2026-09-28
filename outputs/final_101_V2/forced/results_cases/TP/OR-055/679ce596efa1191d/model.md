##### Sets

Let $I = \{1,2,\ldots,14\}$ be the set of display areas (DisplayID from 1 to 14).  
Let $J =$ {Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff} be the set of boat types (ProductName, in source order).

##### Parameters

For each display area $i \in I$:
- $C_i$ = capacity of display area $i$.

For each boat type $j \in J$:
- $v_j$ = value of one unit of boat type $j$.
- $w_j$ = size (weight) of one unit of boat type $j$.

From the retrieved data:

Display area capacities (in source order):
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

Boat types, values, and sizes (in source order):

| $j$                | $v_j$  | $w_j$ |
|--------------------|--------|-------|
| Speedboat          | 69978  | 18    |
| Fishing Boat       | 54011  | 42    |
| Catamaran          | 36352  | 49    |
| Yacht              | 51521  | 42    |
| Sailboat           | 50415  | 41    |
| Kayak              | 76109  | 48    |
| Canoe              | 50462  | 22    |
| Houseboat          | 28989  | 29    |
| Pontoon            | 23318  | 45    |
| Jet Ski            | 26142  | 14    |
| Rowboat            | 42040  | 38    |
| Hovercraft         | 85961  | 47    |
| Cabin Cruiser      | 50142  | 45    |
| Wakeboard Boat     | 48478  | 28    |
| Dinghy             | 60953  | 24    |
| Trawler            | 95265  | 39    |
| Paddle Boat        | 22839  | 32    |
| Submarine          | 90957  | 36    |
| RIB                | 84652  | 14    |
| Skiff              | 78991  | 16    |

##### Decision Variables

For each display area $i \in I$ and boat type $j \in J$:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of boat type $j$ placed in display area $i$.

##### Objective Function

\[
\max \sum_{i=1}^{14} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each display area $i = 1, \ldots, 14$:
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
\max\quad & \sum_{i=1}^{14} \Big(69978\,x_{i,\text{Speedboat}} + 54011\,x_{i,\text{Fishing Boat}} + 36352\,x_{i,\text{Catamaran}} + 51521\,x_{i,\text{Yacht}} + 50415\,x_{i,\text{Sailboat}} \\
&\qquad + 76109\,x_{i,\text{Kayak}} + 50462\,x_{i,\text{Canoe}} + 28989\,x_{i,\text{Houseboat}} + 23318\,x_{i,\text{Pontoon}} + 26142\,x_{i,\text{Jet Ski}} \\
&\qquad + 42040\,x_{i,\text{Rowboat}} + 85961\,x_{i,\text{Hovercraft}} + 50142\,x_{i,\text{Cabin Cruiser}} + 48478\,x_{i,\text{Wakeboard Boat}} \\
&\qquad + 60953\,x_{i,\text{Dinghy}} + 95265\,x_{i,\text{Trawler}} + 22839\,x_{i,\text{Paddle Boat}} + 90957\,x_{i,\text{Submarine}} \\
&\qquad + 84652\,x_{i,\text{RIB}} + 78991\,x_{i,\text{Skiff}} \Big) \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i,\quad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0},\quad \forall i=1,\ldots,14,\ j \in J
\end{align*}
\]

Where for each $i$, $C_i$ is as listed above, and for each $j$, $v_j$ and $w_j$ are as listed above, in source order.