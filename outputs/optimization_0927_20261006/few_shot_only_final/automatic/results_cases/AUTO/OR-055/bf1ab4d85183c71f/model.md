**Sets:**
- $i \in \{\text{1}, \text{2}, \ldots, \text{14}\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of boat type $j$ (from products.csv)
- $w_j$ = Weight of boat type $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of boat type $j$ placed in display area $i$

**Objective:**
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID from 1 to 14):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,14\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,14\},\; \forall j \in \{1,2,\ldots,20\}
\]

**Data:**

From capacity.csv (DisplayID, Capacity):

| DisplayID | Capacity |
|-----------|----------|
| 1         | 356      |
| 2         | 478      |
| 3         | 305      |
| 4         | 291      |
| 5         | 168      |
| 6         | 449      |
| 7         | 139      |
| 8         | 383      |
| 9         | 472      |
| 10        | 288      |
| 11        | 320      |
| 12        | 250      |
| 13        | 402      |
| 14        | 293      |

From products.csv (ProductName, Value, Weight):

| $j$ | ProductName         | Value  | Weight |
|-----|---------------------|--------|--------|
| 1   | Speedboat           | 69978  | 18     |
| 2   | Fishing Boat        | 54011  | 42     |
| 3   | Catamaran           | 36352  | 49     |
| 4   | Yacht               | 51521  | 42     |
| 5   | Sailboat            | 50415  | 41     |
| 6   | Kayak               | 76109  | 48     |
| 7   | Canoe               | 50462  | 22     |
| 8   | Houseboat           | 28989  | 29     |
| 9   | Pontoon             | 23318  | 45     |
| 10  | Jet Ski             | 26142  | 14     |
| 11  | Rowboat             | 42040  | 38     |
| 12  | Hovercraft          | 85961  | 47     |
| 13  | Cabin Cruiser       | 50142  | 45     |
| 14  | Wakeboard Boat      | 48478  | 28     |
| 15  | Dinghy              | 60953  | 24     |
| 16  | Trawler             | 95265  | 39     |
| 17  | Paddle Boat         | 22839  | 32     |
| 18  | Submarine           | 90957  | 36     |
| 19  | RIB                 | 84652  | 14     |
| 20  | Skiff               | 78991  | 16     |

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,14;\; j = 1,\ldots,20
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above, and $c_i$ is the capacity for each DisplayID.