Let $x_{ij}$ be the number of units of boat type $j$ (ProductName) to be placed in display area $i$ (DisplayID). All $x_{ij}$ are nonnegative integers.

Sets:
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}, \text{11}, \text{12}, \text{13}, \text{14}\}$ (DisplayID)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName)

Parameters:
- $v_j$ = Value of boat type $j$ (see table below)
- $w_j$ = Weight (size) of boat type $j$ (see table below)
- $C_i$ = Capacity of display area $i$ (see table below)

Objective:
$$
\max \sum_{i \in \{\text{1},\ldots,\text{14}\}} \sum_{j} v_j \cdot x_{ij}
$$

Subject to (for each $i$):
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{14}\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

#### Data

**Display Areas (from capacity.csv):**

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

**Boat Types (from products.csv):**

| ProductName      | Value  | Weight |
|------------------|--------|--------|
| Speedboat        | 69978  | 18     |
| Fishing Boat     | 54011  | 42     |
| Catamaran        | 36352  | 49     |
| Yacht            | 51521  | 42     |
| Sailboat         | 50415  | 41     |
| Kayak            | 76109  | 48     |
| Canoe            | 50462  | 22     |
| Houseboat        | 28989  | 29     |
| Pontoon          | 23318  | 45     |
| Jet Ski          | 26142  | 14     |
| Rowboat          | 42040  | 38     |
| Hovercraft       | 85961  | 47     |
| Cabin Cruiser    | 50142  | 45     |
| Wakeboard Boat   | 48478  | 28     |
| Dinghy           | 60953  | 24     |
| Trawler          | 95265  | 39     |
| Paddle Boat      | 22839  | 32     |
| Submarine        | 90957  | 36     |
| RIB              | 84652  | 14     |
| Skiff            | 78991  | 16     |

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,14;\ j=1,\ldots,20
\end{align*}
$$

Where $v_j$, $w_j$, and $C_i$ are as given in the tables above, and $x_{ij}$ is the number of units of boat type $j$ in display area $i$.