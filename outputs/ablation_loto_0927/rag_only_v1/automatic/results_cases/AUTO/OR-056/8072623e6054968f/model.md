Sets:
- Let I = {1, 2, ..., 14} be the set of display areas, indexed by i (DisplayID from capacity.csv).
- Let J = {1, 2, ..., 20} be the set of vessel types, indexed by j, corresponding to the order in products.csv:
    1: Speedboat
    2: Fishing Boat
    3: Catamaran
    4: Yacht
    5: Sailboat
    6: Kayak
    7: Canoe
    8: Houseboat
    9: Pontoon
    10: Jet Ski
    11: Rowboat
    12: Hovercraft
    13: Cabin Cruiser
    14: Wakeboard Boat
    15: Dinghy
    16: Trawler
    17: Paddle Boat
    18: Submarine
    19: RIB
    20: Skiff

Parameters:
- Capacity_i: capacity of display area i (from capacity.csv)
- Value_j: value of vessel type j (from products.csv)
- Weight_j: weight (size) of vessel type j (from products.csv)

Variables:
- x_{ij} = number of vessels of type j assigned to display area i, for all i in I, j in J
    Domain: x_{ij} ∈ {0, 1, 2, ...} (nonnegative integers)

Data:
From capacity.csv:
DisplayID | Capacity
1 | 457
2 | 604
3 | 751
4 | 468
5 | 343
6 | 408
7 | 741
8 | 914
9 | 682
10 | 409
11 | 342
12 | 903
13 | 680
14 | 886

From products.csv (ProductName, Value, Weight):
1: Speedboat, 29664, 18
2: Fishing Boat, 31778, 36
3: Catamaran, 73501, 25
4: Yacht, 78255, 16
5: Sailboat, 93606, 97
6: Kayak, 46983, 35
7: Canoe, 95026, 32
8: Houseboat, 57685, 100
9: Pontoon, 60323, 43
10: Jet Ski, 91224, 15
11: Rowboat, 44003, 95
12: Hovercraft, 75998, 57
13: Cabin Cruiser, 84525, 13
14: Wakeboard Boat, 66207, 44
15: Dinghy, 65002, 64
16: Trawler, 33132, 88
17: Paddle Boat, 69239, 42
18: Submarine, 66948, 46
19: RIB, 88240, 24
20: Skiff, 48858, 93

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as above.

Subject to, for each display area i (DisplayID from 1 to 14):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Capacity_i is as above, and Weight_j is as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
\]

Explicitly, for each i = 1,...,14:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
with the following capacities:
- i=1: 457
- i=2: 604
- i=3: 751
- i=4: 468
- i=5: 343
- i=6: 408
- i=7: 741
- i=8: 914
- i=9: 682
- i=10: 409
- i=11: 342
- i=12: 903
- i=13: 680
- i=14: 886

And for each j = 1,...,20, the Value_j and Weight_j are as listed above.

Summary:
- Decision variables: x_{ij} = number of vessels of type j in display area i, integer, ≥0
- Objective: maximize total value of all vessels assigned
- Constraints: for each display area, total weight of assigned vessels ≤ its capacity

This is a complete integer programming formulation using all provided data.