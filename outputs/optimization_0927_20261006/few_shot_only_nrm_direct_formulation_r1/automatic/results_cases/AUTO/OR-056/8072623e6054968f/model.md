**Sets:**
- $i \in \{1,2,\ldots,14\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

**Parameters:**
- $C_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of boat type $j$ (from products.csv)
- $w_j$ = Weight of boat type $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ placed in display area $i$

**Objective:**
\[
\max \sum_{i=1}^{14} \sum_{j} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i$ (DisplayID $i$ from 1 to 14):
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i
\]

For all $i$ and $j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Data:**

*Display Areas and Capacities (capacity.csv):*

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

*Boat Types, Values, and Weights (products.csv):*

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

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{14} \Big( 
29664\, x_{i,\text{Speedboat}} + 
31778\, x_{i,\text{Fishing Boat}} + 
73501\, x_{i,\text{Catamaran}} + 
78255\, x_{i,\text{Yacht}} + 
93606\, x_{i,\text{Sailboat}} + 
46983\, x_{i,\text{Kayak}} + 
95026\, x_{i,\text{Canoe}} + 
57685\, x_{i,\text{Houseboat}} + \\
&\quad
60323\, x_{i,\text{Pontoon}} + 
91224\, x_{i,\text{Jet Ski}} + 
44003\, x_{i,\text{Rowboat}} + 
75998\, x_{i,\text{Hovercraft}} + 
84525\, x_{i,\text{Cabin Cruiser}} + 
66207\, x_{i,\text{Wakeboard Boat}} + 
65002\, x_{i,\text{Dinghy}} + 
33132\, x_{i,\text{Trawler}} + \\
&\quad
69239\, x_{i,\text{Paddle Boat}} + 
66948\, x_{i,\text{Submarine}} + 
88240\, x_{i,\text{RIB}} + 
48858\, x_{i,\text{Skiff}}
\Big)
\end{align*}
\]

Subject to, for each $i = 1, \ldots, 14$ (DisplayID):

\[
\begin{align*}
18\, x_{i,\text{Speedboat}} + 
36\, x_{i,\text{Fishing Boat}} + 
25\, x_{i,\text{Catamaran}} + 
16\, x_{i,\text{Yacht}} + 
97\, x_{i,\text{Sailboat}} + 
35\, x_{i,\text{Kayak}} + 
32\, x_{i,\text{Canoe}} + 
100\, x_{i,\text{Houseboat}} + \\
43\, x_{i,\text{Pontoon}} + 
15\, x_{i,\text{Jet Ski}} + 
95\, x_{i,\text{Rowboat}} + 
57\, x_{i,\text{Hovercraft}} + 
13\, x_{i,\text{Cabin Cruiser}} + 
44\, x_{i,\text{Wakeboard Boat}} + 
64\, x_{i,\text{Dinghy}} + 
88\, x_{i,\text{Trawler}} + \\
42\, x_{i,\text{Paddle Boat}} + 
46\, x_{i,\text{Submarine}} + 
24\, x_{i,\text{RIB}} + 
93\, x_{i,\text{Skiff}}
\leq C_i
\end{align*}
\]

where $C_i$ is the capacity for DisplayID $i$ as listed above.

And for all $i = 1,\ldots,14$ and all $j$ (20 boat types):

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]