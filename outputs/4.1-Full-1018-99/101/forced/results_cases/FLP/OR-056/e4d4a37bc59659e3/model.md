##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ to be placed in display area $i$.

Where:
- $i \in D = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ (Display areas)
- $j \in P =$ 
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

Vessel types, values, and weights:

| Product Name      | Value  | Weight |
|-------------------|--------|--------|
| Speedboat         | 29664  | 18     |
| Fishing Boat      | 31778  | 36     |
| Catamaran         | 73501  | 25     |
| Yacht             | 78255  | 16     |
| Sailboat          | 93606  | 97     |
| Kayak             | 46983  | 35     |
| Canoe             | 95026  | 32     |
| Houseboat         | 57685  | 100    |
| Pontoon           | 60323  | 43     |
| Jet Ski           | 91224  | 15     |
| Rowboat           | 44003  | 95     |
| Hovercraft        | 75998  | 57     |
| Cabin Cruiser     | 84525  | 13     |
| Wakeboard Boat    | 66207  | 44     |
| Dinghy            | 65002  | 64     |
| Trawler           | 33132  | 88     |
| Paddle Boat       | 69239  | 42     |
| Submarine         | 66948  | 46     |
| RIB               | 88240  | 24     |
| Skiff             | 48858  | 93     |

Let $v_j$ be the value and $w_j$ the weight of vessel type $j$.

##### Objective Function

\[
\max \sum_{i \in D} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. **Capacity constraints for each display area:**
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in D
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in D,\, j \in P
   \]

##### Complete Parameter Listing

- $D = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$
- $C = [457, 604, 751, 468, 343, 408, 741, 914, 682, 409, 342, 903, 680, 886]$
- $P = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$
- $v = [29664, 31778, 73501, 78255, 93606, 46983, 95026, 57685, 60323, 91224, 44003, 75998, 84525, 66207, 65002, 33132, 69239, 66948, 88240, 48858]$
- $w = [18, 36, 25, 16, 97, 35, 32, 100, 43, 15, 95, 57, 13, 44, 64, 88, 42, 46, 24, 93]$

##### Model Summary

Maximize the total value of boats assigned to display areas, subject to each area's capacity, by choosing nonnegative integer numbers of each vessel type per area.