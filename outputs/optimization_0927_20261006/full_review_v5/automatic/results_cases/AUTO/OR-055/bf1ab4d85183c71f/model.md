Let $x_{ij}$ be the number of units of boat type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of one unit of boat type $j$ (from products.csv)
- $W_j$: Weight (size) of one unit of boat type $j$ (from products.csv)
- $C_i$: Capacity of display area $i$ (from capacity.csv)

**Sets:**

- $i \in \{1,2,\ldots,14\}$ (DisplayID from capacity.csv, in source order)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv, in source order)

---

### Mathematical Model

**Objective:**

\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} V_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID):

\[
\sum_{j=1}^{20} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,14\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,14\},\; \forall j \in \{1,2,\ldots,20\}
\]

---

#### Parameter Tables (source order):

**Display Areas (capacity.csv):**

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

**Boat Types (products.csv):**

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

**Decision variables:**

\[
x_{ij} = \text{number of units of boat type } j \text{ placed in display area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Summary:**

Maximize total value of boats allocated to display areas, subject to each area's capacity, with integer numbers of each boat type per area. All parameters and identifiers are as retrieved and in source order.