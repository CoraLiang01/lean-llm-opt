Let $x_{ij}$ be the number of units of boat type $j$ (ProductName) to be placed in display area $i$ (DisplayID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $D$ be the set of display areas, indexed by $i$ (DisplayID from capacity.csv).
- Let $P$ be the set of boat types, indexed by $j$ (ProductName from products.csv).
- $C_i$ = Capacity of display area $i$.
- $v_j$ = Value of one unit of boat type $j$.
- $w_j$ = Weight (size) of one unit of boat type $j$.

**Data:**

Display Areas (from capacity.csv, in source order):

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

Boat Types (from products.csv, in source order):

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

---

### Mathematical Model

**Objective:**

\[
\max \sum_{i \in D} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i \in D$ (DisplayID):

\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in D$, $j \in P$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $D = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$
- $P =$ (in source order): Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff
- $C_i$ as given above for each $i$
- $v_j$, $w_j$ as given above for each $j$

---

**Decision variables:**

- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$, integer, $\geq 0$