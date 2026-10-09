Let $x_{ij}$ = number of units of boat type $j$ placed in display area $i$.

Indices:
- $i$ indexes display areas, with DisplayID from capacity.csv: $i \in \{1,2,\ldots,14\}$
- $j$ indexes boat types, with ProductName from products.csv: $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

Parameters:
- $v_j$ = Value of boat type $j$ (from products.csv)
- $w_j$ = Weight of boat type $j$ (from products.csv)
- $C_i$ = Capacity of display area $i$ (from capacity.csv)

**Model:**

Maximize total value:
$$
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

Subject to, for each display area $i$ (DisplayID as below):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,14\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Where:**

From capacity.csv (in source order):

| DisplayID | $C_i$ |
|-----------|-------|
| 1         | 356   |
| 2         | 478   |
| 3         | 305   |
| 4         | 291   |
| 5         | 168   |
| 6         | 449   |
| 7         | 139   |
| 8         | 383   |
| 9         | 472   |
| 10        | 288   |
| 11        | 320   |
| 12        | 250   |
| 13        | 402   |
| 14        | 293   |

From products.csv (in source order):

| ProductName         | $v_j$ | $w_j$ |
|---------------------|-------|-------|
| Speedboat           | 69978 | 18    |
| Fishing Boat        | 54011 | 42    |
| Catamaran           | 36352 | 49    |
| Yacht               | 51521 | 42    |
| Sailboat            | 50415 | 41    |
| Kayak               | 76109 | 48    |
| Canoe               | 50462 | 22    |
| Houseboat           | 28989 | 29    |
| Pontoon             | 23318 | 45    |
| Jet Ski             | 26142 | 14    |
| Rowboat             | 42040 | 38    |
| Hovercraft          | 85961 | 47    |
| Cabin Cruiser       | 50142 | 45    |
| Wakeboard Boat      | 48478 | 28    |
| Dinghy              | 60953 | 24    |
| Trawler             | 95265 | 39    |
| Paddle Boat         | 22839 | 32    |
| Submarine           | 90957 | 36    |
| RIB                 | 84652 | 14    |
| Skiff               | 78991 | 16    |

**Decision variables:**
$$
x_{ij} \in \{0, 1, 2, \ldots\} \qquad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
$$