**Sets:**

- Let $I$ be the set of display areas, indexed by $i$, with DisplayID from capacity.csv:
  $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$
- Let $J$ be the set of vessel types, indexed by $j$, with ProductName from products.csv:
  $J = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

**Parameters:**

- $c_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of vessel type $j$ (from products.csv)
- $w_j$ = Weight (size) of vessel type $j$ (from products.csv)

**Decision Variables:**

- $x_{ij}$ = number of vessels of type $j$ to be placed in display area $i$  
  ($x_{ij} \in \mathbb{Z}_{\geq 0}$, i.e., nonnegative integers)

---

### Mathematical Model

**Objective:**

\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i \in I$:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
\]

For all $i \in I$, $j \in J$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data

**Display Areas and Capacities (from capacity.csv):**

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

**Vessel Types, Values, and Weights (from products.csv):**

| ProductName      | Value  | Weight |
|------------------|--------|--------|
| Speedboat        | 29664  | 18     |
| Fishing Boat     | 31778  | 36     |
| Catamaran        | 73501  | 25     |
| Yacht            | 78255  | 16     |
| Sailboat         | 93606  | 97     |
| Kayak            | 46983  | 35     |
| Canoe            | 95026  | 32     |
| Houseboat        | 57685  | 100    |
| Pontoon          | 60323  | 43     |
| Jet Ski          | 91224  | 15     |
| Rowboat          | 44003  | 95     |
| Hovercraft       | 75998  | 57     |
| Cabin Cruiser    | 84525  | 13     |
| Wakeboard Boat   | 66207  | 44     |
| Dinghy           | 65002  | 64     |
| Trawler          | 33132  | 88     |
| Paddle Boat      | 69239  | 42     |
| Submarine        | 66948  | 46     |
| RIB              | 88240  | 24     |
| Skiff            | 48858  | 93     |

---

**Complete Formulation:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i, \quad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,14,\; j = 1,\ldots,20
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above for each vessel type $j$, and $c_i$ is the capacity for each display area $i$.