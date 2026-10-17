Let $x_{ij}$ be the number of units of boat type $j$ to be placed in display area $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $I$ be the set of display areas (indexed by DisplayID), and $J$ be the set of boat types (indexed by ProductName).

Let $c_i$ be the capacity of display area $i$ (from "capacity.csv").

Let $v_j$ be the value and $w_j$ the weight (size) of boat type $j$ (from "products.csv").

**Sets:**

- $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$
- $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

**Parameters:**

From capacity.csv:

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

From products.csv:

| ProductName      | Value | Weight |
|------------------|-------|--------|
| Speedboat        | 69978 | 18     |
| Fishing Boat     | 54011 | 42     |
| Catamaran        | 36352 | 49     |
| Yacht            | 51521 | 42     |
| Sailboat         | 50415 | 41     |
| Kayak            | 76109 | 48     |
| Canoe            | 50462 | 22     |
| Houseboat        | 28989 | 29     |
| Pontoon          | 23318 | 45     |
| Jet Ski          | 26142 | 14     |
| Rowboat          | 42040 | 38     |
| Hovercraft       | 85961 | 47     |
| Cabin Cruiser    | 50142 | 45     |
| Wakeboard Boat   | 48478 | 28     |
| Dinghy           | 60953 | 24     |
| Trawler          | 95265 | 39     |
| Paddle Boat      | 22839 | 32     |
| Submarine        | 90957 | 36     |
| RIB              | 84652 | 14     |
| Skiff            | 78991 | 16     |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

Capacity constraints for each display area:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]

Non-negativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

**Where:**

- $x_{ij}$: Number of units of boat type $j$ in display area $i$ (integer, $\geq 0$)
- $v_j$: Value of boat type $j$ (see table above)
- $w_j$: Weight (size) of boat type $j$ (see table above)
- $c_i$: Capacity of display area $i$ (see table above)

**All identifiers and coefficients are as retrieved above.**