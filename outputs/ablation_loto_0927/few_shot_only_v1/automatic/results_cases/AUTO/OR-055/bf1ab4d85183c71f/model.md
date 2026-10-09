**Mathematical Optimization Model**

**Sets:**
- Let $I$ be the set of display areas, indexed by $i$, with DisplayID from capacity.csv.
- Let $J$ be the set of boat types, indexed by $j$, with ProductName from products.csv.

**Parameters:**
- $C_i$: Capacity of display area $i$ (from capacity.csv).
- $v_j$: Value of one unit of boat type $j$ (from products.csv).
- $w_j$: Size (Weight) of one unit of boat type $j$ (from products.csv).

**Decision Variables:**
- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$.
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers).

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

---

**Subject to:**

**1. Capacity constraints for each display area:**
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

**2. Integrality and nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

**Parameter Data (from CSVs):**

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

| ProductName        | Value  | Weight |
|--------------------|--------|--------|
| Speedboat          | 69978  | 18     |
| Fishing Boat       | 54011  | 42     |
| Catamaran          | 36352  | 49     |
| Yacht              | 51521  | 42     |
| Sailboat           | 50415  | 41     |
| Kayak              | 76109  | 48     |
| Canoe              | 50462  | 22     |
| Houseboat          | 28989  | 29     |
| Pontoon            | 23318  | 45     |
| Jet Ski            | 26142  | 14     |
| Rowboat            | 42040  | 38     |
| Hovercraft         | 85961  | 47     |
| Cabin Cruiser      | 50142  | 45     |
| Wakeboard Boat     | 48478  | 28     |
| Dinghy             | 60953  | 24     |
| Trawler            | 95265  | 39     |
| Paddle Boat        | 22839  | 32     |
| Submarine          | 90957  | 36     |
| RIB                | 84652  | 14     |
| Skiff              | 78991  | 16     |

---

**Full Model (with explicit indices):**

**Variables:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,14\},\, j \in \{\text{Speedboat}, \ldots, \text{Skiff}\}
\]

**Objective:**
\[
\max \sum_{i=1}^{14} \Bigg(
69978\,x_{i,\text{Speedboat}} +
54011\,x_{i,\text{Fishing Boat}} +
36352\,x_{i,\text{Catamaran}} +
51521\,x_{i,\text{Yacht}} +
50415\,x_{i,\text{Sailboat}} +
76109\,x_{i,\text{Kayak}} +
50462\,x_{i,\text{Canoe}} +
28989\,x_{i,\text{Houseboat}} +
23318\,x_{i,\text{Pontoon}} +
26142\,x_{i,\text{Jet Ski}} +
42040\,x_{i,\text{Rowboat}} +
85961\,x_{i,\text{Hovercraft}} +
50142\,x_{i,\text{Cabin Cruiser}} +
48478\,x_{i,\text{Wakeboard Boat}} +
60953\,x_{i,\text{Dinghy}} +
95265\,x_{i,\text{Trawler}} +
22839\,x_{i,\text{Paddle Boat}} +
90957\,x_{i,\text{Submarine}} +
84652\,x_{i,\text{RIB}} +
78991\,x_{i,\text{Skiff}}
\Bigg)
\]

**Subject to, for each $i$ (DisplayID):**
\[
\begin{align*}
&18\,x_{i,\text{Speedboat}} + 42\,x_{i,\text{Fishing Boat}} + 49\,x_{i,\text{Catamaran}} + 42\,x_{i,\text{Yacht}} + 41\,x_{i,\text{Sailboat}} + 48\,x_{i,\text{Kayak}} + 22\,x_{i,\text{Canoe}} + 29\,x_{i,\text{Houseboat}} \\
&+ 45\,x_{i,\text{Pontoon}} + 14\,x_{i,\text{Jet Ski}} + 38\,x_{i,\text{Rowboat}} + 47\,x_{i,\text{Hovercraft}} + 45\,x_{i,\text{Cabin Cruiser}} + 28\,x_{i,\text{Wakeboard Boat}} + 24\,x_{i,\text{Dinghy}} \\
&+ 39\,x_{i,\text{Trawler}} + 32\,x_{i,\text{Paddle Boat}} + 36\,x_{i,\text{Submarine}} + 14\,x_{i,\text{RIB}} + 16\,x_{i,\text{Skiff}} \leq C_i
\end{align*}
\]
where $C_i$ is the capacity for DisplayID $i$ as given above.

**And:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,\,j
\]