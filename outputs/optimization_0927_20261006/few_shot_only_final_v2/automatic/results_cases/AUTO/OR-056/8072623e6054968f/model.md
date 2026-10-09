Let  
- $i$ index display areas, with $i \in \{\text{1}, \text{2}, \ldots, \text{14}\}$ (from "DisplayID" in capacity.csv)  
- $j$ index vessel types, with $j \in \{\text{Speedboat}, \text{Fishing Boat}, \text{Catamaran}, \text{Yacht}, \text{Sailboat}, \text{Kayak}, \text{Canoe}, \text{Houseboat}, \text{Pontoon}, \text{Jet Ski}, \text{Rowboat}, \text{Hovercraft}, \text{Cabin Cruiser}, \text{Wakeboard Boat}, \text{Dinghy}, \text{Trawler}, \text{Paddle Boat}, \text{Submarine}, \text{RIB}, \text{Skiff}\}$ (from "ProductName" in products.csv)

Parameters:  
- $v_j$ = value of vessel type $j$ (from "Value" in products.csv)  
- $w_j$ = weight of vessel type $j$ (from "Weight" in products.csv)  
- $C_i$ = capacity of display area $i$ (from "Capacity" in capacity.csv)

Decision variables:  
- $x_{ij}$ = number of vessels of type $j$ to place in display area $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Model:

**Objective:**  
$$
\max \sum_{i \in \{\text{1},\ldots,\text{14}\}} \sum_{j \in \{\text{Speedboat}, \ldots, \text{Skiff}\}} v_j \cdot x_{ij}
$$

**Subject to:**  

For each display area $i$ (DisplayID from 1 to 14):  
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i
$$

**Variable domains:**  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

**Data:**  

From capacity.csv:  
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

From products.csv:  
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

$$
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,14,\; j = 1,\ldots,20
\end{align*}
$$

Where the mapping of $j$ to vessel types, $v_j$, and $w_j$ is as per the tables above, and $C_i$ is as per the capacity table.