**Sets and Indices:**
- $i \in \{\text{1, 2, ..., 14}\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

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
| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Speedboat          | 69978 | 18     |
| Fishing Boat       | 54011 | 42     |
| Catamaran          | 36352 | 49     |
| Yacht              | 51521 | 42     |
| Sailboat           | 50415 | 41     |
| Kayak              | 76109 | 48     |
| Canoe              | 50462 | 22     |
| Houseboat          | 28989 | 29     |
| Pontoon            | 23318 | 45     |
| Jet Ski            | 26142 | 14     |
| Rowboat            | 42040 | 38     |
| Hovercraft         | 85961 | 47     |
| Cabin Cruiser      | 50142 | 45     |
| Wakeboard Boat     | 48478 | 28     |
| Dinghy             | 60953 | 24     |
| Trawler            | 95265 | 39     |
| Paddle Boat        | 22839 | 32     |
| Submarine          | 90957 | 36     |
| RIB                | 84652 | 14     |
| Skiff              | 78991 | 16     |

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all display areas $i$ and boat types $j$

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of boat type $j$.

**Constraints:**

For each display area $i$ (DisplayID $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,14\}
\]
where $w_j$ is the weight of boat type $j$, and $c_i$ is the capacity of display area $i$.

**Variable Domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Explicit Data:**

- Display areas $i$ (DisplayID): 1, 2, ..., 14, with capacities as above.
- Boat types $j$ (ProductName): as listed above, with corresponding Value and Weight.

**Full Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1, \ldots, 14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 14;\; j = 1, \ldots, 20
\end{align*}
\]

Where:

- For $j=1$ (Speedboat): $v_1=69978$, $w_1=18$
- For $j=2$ (Fishing Boat): $v_2=54011$, $w_2=42$
- For $j=3$ (Catamaran): $v_3=36352$, $w_3=49$
- For $j=4$ (Yacht): $v_4=51521$, $w_4=42$
- For $j=5$ (Sailboat): $v_5=50415$, $w_5=41$
- For $j=6$ (Kayak): $v_6=76109$, $w_6=48$
- For $j=7$ (Canoe): $v_7=50462$, $w_7=22$
- For $j=8$ (Houseboat): $v_8=28989$, $w_8=29$
- For $j=9$ (Pontoon): $v_9=23318$, $w_9=45$
- For $j=10$ (Jet Ski): $v_{10}=26142$, $w_{10}=14$
- For $j=11$ (Rowboat): $v_{11}=42040$, $w_{11}=38$
- For $j=12$ (Hovercraft): $v_{12}=85961$, $w_{12}=47$
- For $j=13$ (Cabin Cruiser): $v_{13}=50142$, $w_{13}=45$
- For $j=14$ (Wakeboard Boat): $v_{14}=48478$, $w_{14}=28$
- For $j=15$ (Dinghy): $v_{15}=60953$, $w_{15}=24$
- For $j=16$ (Trawler): $v_{16}=95265$, $w_{16}=39$
- For $j=17$ (Paddle Boat): $v_{17}=22839$, $w_{17}=32$
- For $j=18$ (Submarine): $v_{18}=90957$, $w_{18}=36$
- For $j=19$ (RIB): $v_{19}=84652$, $w_{19}=14$
- For $j=20$ (Skiff): $v_{20}=78991$, $w_{20}=16$

And for each display area $i$ (DisplayID), $c_i$ is as in the table above.

**Summary:**  
Maximize the total value of boats allocated to display areas, subject to each area's capacity, with integer numbers of each boat type per area.