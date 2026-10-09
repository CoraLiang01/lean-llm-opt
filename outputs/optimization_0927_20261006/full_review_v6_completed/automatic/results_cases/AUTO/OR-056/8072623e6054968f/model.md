Let $x_{ij}$ be the number of vessels of type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of vessel type $j$ (from 'products.csv')
- $W_j$: Weight (size) of vessel type $j$ (from 'products.csv')
- $C_i$: Capacity of display area $i$ (from 'capacity.csv')

**Sets:**

- $i \in \{\text{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}\}$ (DisplayID)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName)

**Data:**

Display Areas (from 'capacity.csv'):
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

Vessel Types (from 'products.csv'):
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

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\},\ j \in \{\text{all ProductNames above}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{14} \sum_{j} V_j \cdot x_{ij}
$$

**Subject to:**

For each display area $i$:
$$
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,14\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Where:**

- $V_j$ and $W_j$ are as given in the table above for each ProductName $j$.
- $C_i$ is as given in the table above for each DisplayID $i$.

---

**All data and identifiers are preserved as retrieved.**