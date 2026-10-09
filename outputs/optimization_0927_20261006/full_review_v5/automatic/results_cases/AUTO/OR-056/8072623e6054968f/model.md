Let $x_{ij}$ be the number of vessels of type $j$ (ProductName) to be placed in display area $i$ (DisplayID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $D$ = set of display areas (DisplayID from capacity.csv)
- $P$ = set of vessel types (ProductName from products.csv)
- $C_i$ = capacity of display area $i$ (Capacity from capacity.csv)
- $v_j$ = value of vessel type $j$ (Value from products.csv)
- $w_j$ = size of vessel type $j$ (Weight from products.csv)

---

### Objective

$$
\max \sum_{i \in D} \sum_{j \in P} v_j \cdot x_{ij}
$$

### Constraints

For each display area $i \in D$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in D$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

### Parameter Tables

#### Display Areas (capacity.csv, in source order)

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

#### Vessel Types (products.csv, in source order)

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

**Decision variables:**  
$x_{ij}$: number of vessels of type $j$ to be placed in display area $i$, for all $i \in D$, $j \in P$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Model summary:**  
Maximize total value of vessels assigned to displays, subject to each display's capacity, using the above parameters and constraints.