**Sets:**
- $i \in \{\text{1}, \text{2}, \ldots, \text{14}\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

**Parameters:**
- $C_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of boat type $j$ (from products.csv)
- $w_j$ = Weight of boat type $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of boats of type $j$ placed in display area $i$

**Objective:**
\[
\max \sum_{i \in \{\text{1},\ldots,14\}} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID from capacity.csv):

\[
\sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,14\}
\]

For all $i, j$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Numerical Data:**

*Display Areas and Capacities (from capacity.csv, in source order):*

| DisplayID | Capacity |
|-----------|----------|
| 1         | 457      |
| 2         | 604      |
| 3         | 751      |
| 4         | 468      |
| 5         | 343      |
| 6         | 408      |
| 7         | 741      |
| 8         | 914      |
| 9         | 682      |
| 10        | 409      |
| 11        | 342      |
| 12        | 903      |
| 13        | 680      |
| 14        | 886      |

*Boat Types, Values, and Weights (from products.csv, in source order):*

| ProductName        | Value  | Weight |
|--------------------|--------|--------|
| Speedboat          | 29664  | 18     |
| Fishing Boat       | 31778  | 36     |
| Catamaran          | 73501  | 25     |
| Yacht              | 78255  | 16     |
| Sailboat           | 93606  | 97     |
| Kayak              | 46983  | 35     |
| Canoe              | 95026  | 32     |
| Houseboat          | 57685  | 100    |
| Pontoon            | 60323  | 43     |
| Jet Ski            | 91224  | 15     |
| Rowboat            | 44003  | 95     |
| Hovercraft         | 75998  | 57     |
| Cabin Cruiser      | 84525  | 13     |
| Wakeboard Boat     | 66207  | 44     |
| Dinghy             | 65002  | 64     |
| Trawler            | 33132  | 88     |
| Paddle Boat        | 69239  | 42     |
| Submarine          | 66948  | 46     |
| RIB                | 88240  | 24     |
| Skiff              | 48858  | 93     |

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,14;\ j = 1,\ldots,20
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above, and $C_i$ is the capacity for DisplayID $i$ as listed above.