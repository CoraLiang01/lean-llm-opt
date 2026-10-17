##### Sets

Let $I$ be the set of display areas (showcases), indexed by $i$.

Let $J$ be the set of vessel types (products), indexed by $j$.

##### Parameters

- $C_i$: Capacity of display area $i$.

- $v_j$: Value of vessel type $j$.

- $w_j$: Size (weight/dimension) of vessel type $j$.

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ placed in display area $i$.

##### Objective Function

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

##### Constraints

For each display area $i \in I$:

$\sum_{j \in J} w_j x_{ij} \leq C_i$

For all $i \in I$, $j \in J$:

$x_{ij} \geq 0$ and integer

---

##### Retrieved Information

**Display Areas and Capacities:**

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

So, $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$

$C_1 = 457$, $C_2 = 604$, $C_3 = 751$, $C_4 = 468$, $C_5 = 343$, $C_6 = 408$, $C_7 = 741$, $C_8 = 914$, $C_9 = 682$, $C_{10} = 409$, $C_{11} = 342$, $C_{12} = 903$, $C_{13} = 680$, $C_{14} = 886$

**Vessel Types, Values, and Sizes:**

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

So, $J$ is the set of all above product names, with $v_j$ and $w_j$ as listed.

##### Full Model

$\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}$

Subject to, for each $i \in \{1,\ldots,14\}$:

$\sum_{j=1}^{20} w_j x_{ij} \leq C_i$

For all $i \in \{1,\ldots,14\}$, $j \in \{1,\ldots,20\}$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$

##### Parameter Values

- $C = [457, 604, 751, 468, 343, 408, 741, 914, 682, 409, 342, 903, 680, 886]$
- $J = [\text{Speedboat}, \text{Fishing Boat}, \text{Catamaran}, \text{Yacht}, \text{Sailboat}, \text{Kayak}, \text{Canoe}, \text{Houseboat}, \text{Pontoon}, \text{Jet Ski}, \text{Rowboat}, \text{Hovercraft}, \text{Cabin Cruiser}, \text{Wakeboard Boat}, \text{Dinghy}, \text{Trawler}, \text{Paddle Boat}, \text{Submarine}, \text{RIB}, \text{Skiff}]$
- $v = [29664, 31778, 73501, 78255, 93606, 46983, 95026, 57685, 60323, 91224, 44003, 75998, 84525, 66207, 65002, 33132, 69239, 66948, 88240, 48858]$
- $w = [18, 36, 25, 16, 97, 35, 32, 100, 43, 15, 95, 57, 13, 44, 64, 88, 42, 46, 24, 93]$