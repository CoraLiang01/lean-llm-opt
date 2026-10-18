##### Sets

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$: Display areas (DisplayID from capacity.csv)
- $J = \{$Speedboat, Fishing\ Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet\ Ski, Rowboat, Hovercraft, Cabin\ Cruiser, Wakeboard\ Boat, Dinghy, Trawler, Paddle\ Boat, Submarine, RIB, Skiff$\}$: Vessel types (ProductName from products.csv)

##### Parameters

- $C_i$: Capacity of display area $i$ (from capacity.csv)
- $v_j$: Value of vessel type $j$ (from products.csv)
- $w_j$: Size (Weight) of vessel type $j$ (from products.csv)

Capacities (in source order):

\[
\begin{align*}
C_1 &= 457 \\
C_2 &= 604 \\
C_3 &= 751 \\
C_4 &= 468 \\
C_5 &= 343 \\
C_6 &= 408 \\
C_7 &= 741 \\
C_8 &= 914 \\
C_9 &= 682 \\
C_{10} &= 409 \\
C_{11} &= 342 \\
C_{12} &= 903 \\
C_{13} &= 680 \\
C_{14} &= 886 \\
\end{align*}
\]

Vessel types, values, and sizes (in source order):

| $j$                | $v_j$  | $w_j$ |
|--------------------|--------|-------|
| Speedboat          | 29664  | 18    |
| Fishing Boat       | 31778  | 36    |
| Catamaran          | 73501  | 25    |
| Yacht              | 78255  | 16    |
| Sailboat           | 93606  | 97    |
| Kayak              | 46983  | 35    |
| Canoe              | 95026  | 32    |
| Houseboat          | 57685  | 100   |
| Pontoon            | 60323  | 43    |
| Jet Ski            | 91224  | 15    |
| Rowboat            | 44003  | 95    |
| Hovercraft         | 75998  | 57    |
| Cabin Cruiser      | 84525  | 13    |
| Wakeboard Boat     | 66207  | 44    |
| Dinghy             | 65002  | 64    |
| Trawler            | 33132  | 88    |
| Paddle Boat        | 69239  | 42    |
| Submarine          | 66948  | 46    |
| RIB                | 88240  | 24    |
| Skiff              | 48858  | 93    |

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ placed in display area $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. Capacity constraints for each display area:
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
   \]
2. Non-negativity and integrality:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

Display areas and capacities (in source order):

- 1: 457
- 2: 604
- 3: 751
- 4: 468
- 5: 343
- 6: 408
- 7: 741
- 8: 914
- 9: 682
- 10: 409
- 11: 342
- 12: 903
- 13: 680
- 14: 886

Vessel types, values, and sizes (in source order):

- Speedboat: Value 29664, Weight 18
- Fishing Boat: Value 31778, Weight 36
- Catamaran: Value 73501, Weight 25
- Yacht: Value 78255, Weight 16
- Sailboat: Value 93606, Weight 97
- Kayak: Value 46983, Weight 35
- Canoe: Value 95026, Weight 32
- Houseboat: Value 57685, Weight 100
- Pontoon: Value 60323, Weight 43
- Jet Ski: Value 91224, Weight 15
- Rowboat: Value 44003, Weight 95
- Hovercraft: Value 75998, Weight 57
- Cabin Cruiser: Value 84525, Weight 13
- Wakeboard Boat: Value 66207, Weight 44
- Dinghy: Value 65002, Weight 64
- Trawler: Value 33132, Weight 88
- Paddle Boat: Value 69239, Weight 42
- Submarine: Value 66948, Weight 46
- RIB: Value 88240, Weight 24
- Skiff: Value 48858, Weight 93