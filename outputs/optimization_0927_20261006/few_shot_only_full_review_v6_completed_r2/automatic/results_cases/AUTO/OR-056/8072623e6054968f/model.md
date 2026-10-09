**Sets:**
- $i \in \{\text{DisplayID } 1, 2, \ldots, 14\}$ (display areas, as in capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (vessel types, as in products.csv)

**Parameters:**
- $c_i$ = Capacity of display area $i$ (from capacity.csv)
- $v_j$ = Value of vessel type $j$ (from products.csv)
- $w_j$ = Weight (size) of vessel type $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ placed in display area $i$

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i \in \{\text{DisplayID } 1,\ldots,14\}} \sum_{j \in \{\text{all ProductName}\}} v_j \cdot x_{ij}
$$

Subject to, for each display area $i$:
$$
\sum_{j \in \{\text{all ProductName}\}} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{DisplayID } 1,\ldots,14\}
$$

And for all $i, j$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Where:**

From capacity.csv (DisplayID, Capacity):

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

From products.csv (ProductName, Value, Weight):

| ProductName         | Value  | Weight |
|---------------------|--------|--------|
| Speedboat           | 29664  | 18     |
| Fishing Boat        | 31778  | 36     |
| Catamaran           | 73501  | 25     |
| Yacht               | 78255  | 16     |
| Sailboat            | 93606  | 97     |
| Kayak               | 46983  | 35     |
| Canoe               | 95026  | 32     |
| Houseboat           | 57685  | 100    |
| Pontoon             | 60323  | 43     |
| Jet Ski             | 91224  | 15     |
| Rowboat             | 44003  | 95     |
| Hovercraft          | 75998  | 57     |
| Cabin Cruiser       | 84525  | 13     |
| Wakeboard Boat      | 66207  | 44     |
| Dinghy              | 65002  | 64     |
| Trawler             | 33132  | 88     |
| Paddle Boat         | 69239  | 42     |
| Submarine           | 66948  | 46     |
| RIB                 | 88240  | 24     |
| Skiff               | 48858  | 93     |

**Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i$ (DisplayID 1–14), $j$ (all ProductName above).