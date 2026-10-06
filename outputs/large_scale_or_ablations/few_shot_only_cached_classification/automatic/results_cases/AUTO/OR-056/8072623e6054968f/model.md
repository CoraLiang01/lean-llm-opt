**Sets and Indices:**

- Let $I$ be the set of display areas, indexed by $i$ (DisplayID from capacity.csv).
- Let $J$ be the set of vessel types, indexed by $j$ (ProductName from products.csv).

**Parameters:**

- $C_i$: Capacity of display area $i$ (from capacity.csv).
- $v_j$: Value of vessel type $j$ (from products.csv).
- $w_j$: Weight (size) of vessel type $j$ (from products.csv).

**Decision Variables:**

- $x_{ij}$: Number of vessels of type $j$ to be placed in display area $i$.
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers).

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID):

\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

**Parameter Data (as retrieved):**

*Display Areas (capacity.csv):*

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

*Vessel Types (products.csv):*

| ProductName      | Value | Weight |
|------------------|-------|--------|
| Speedboat        | 29664 | 18     |
| Fishing Boat     | 31778 | 36     |
| Catamaran        | 73501 | 25     |
| Yacht            | 78255 | 16     |
| Sailboat         | 93606 | 97     |
| Kayak            | 46983 | 35     |
| Canoe            | 95026 | 32     |
| Houseboat        | 57685 | 100    |
| Pontoon          | 60323 | 43     |
| Jet Ski          | 91224 | 15     |
| Rowboat          | 44003 | 95     |
| Hovercraft       | 75998 | 57     |
| Cabin Cruiser    | 84525 | 13     |
| Wakeboard Boat   | 66207 | 44     |
| Dinghy           | 65002 | 64     |
| Trawler          | 33132 | 88     |
| Paddle Boat      | 69239 | 42     |
| Submarine        | 66948 | 46     |
| RIB              | 88240 | 24     |
| Skiff            | 48858 | 93     |

---

**Summary:**

Maximize the total value of boats assigned to display areas, subject to each area's capacity, with integer numbers of each vessel type per area, using the above data and constraints.