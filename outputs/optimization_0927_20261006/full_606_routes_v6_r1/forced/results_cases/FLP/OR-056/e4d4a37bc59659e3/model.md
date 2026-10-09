##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ to be placed in display area $i$.

Where:
- $i \in I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ (Display areas)
- $j \in J =$ 
  {Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff}

##### Parameters

Display area capacities:
- $C_1 = 457$
- $C_2 = 604$
- $C_3 = 751$
- $C_4 = 468$
- $C_5 = 343$
- $C_6 = 408$
- $C_7 = 741$
- $C_8 = 914$
- $C_9 = 682$
- $C_{10} = 409$
- $C_{11} = 342$
- $C_{12} = 903$
- $C_{13} = 680$
- $C_{14} = 886$

Vessel types, values, and sizes:

| $j$              | Value ($v_j$) | Size ($w_j$) |
|------------------|--------------|-------------|
| Speedboat        | 29664        | 18          |
| Fishing Boat     | 31778        | 36          |
| Catamaran        | 73501        | 25          |
| Yacht            | 78255        | 16          |
| Sailboat         | 93606        | 97          |
| Kayak            | 46983        | 35          |
| Canoe            | 95026        | 32          |
| Houseboat        | 57685        | 100         |
| Pontoon          | 60323        | 43          |
| Jet Ski          | 91224        | 15          |
| Rowboat          | 44003        | 95          |
| Hovercraft       | 75998        | 57          |
| Cabin Cruiser    | 84525        | 13          |
| Wakeboard Boat   | 66207        | 44          |
| Dinghy           | 65002        | 64          |
| Trawler          | 33132        | 88          |
| Paddle Boat      | 69239        | 42          |
| Submarine        | 66948        | 46          |
| RIB              | 88240        | 24          |
| Skiff            | 48858        | 93          |

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Display area capacity constraints:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

###### Retrieved Information

- Display areas: $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$
- Capacities: $C = [457, 604, 751, 468, 343, 408, 741, 914, 682, 409, 342, 903, 680, 886]$
- Vessel types: $J =$ {Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff}
- Values: $v_j$ and Sizes: $w_j$ as listed above.