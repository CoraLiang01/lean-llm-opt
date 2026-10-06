Sets and Indices:
- Let I = {1, 2, ..., 14} be the set of display areas, indexed by i (DisplayID from capacity.csv).
- Let J = {1, 2, ..., 20} be the set of vessel types, indexed by j (row order from products.csv).

Parameters:
- Capacity_i: Capacity of display area i (from capacity.csv).
- Value_j: Value of vessel type j (from products.csv).
- Weight_j: Weight (size) of vessel type j (from products.csv).

Decision Variables:
- x_{ij} = number of vessels of type j to be placed in display area i (integer, x_{ij} ≥ 0).

Data (in source order):

capacity.csv

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

products.csv

| j | ProductName      | Value  | Weight |
|---|------------------|--------|--------|
| 1 | Speedboat        | 29664  | 18     |
| 2 | Fishing Boat     | 31778  | 36     |
| 3 | Catamaran        | 73501  | 25     |
| 4 | Yacht            | 78255  | 16     |
| 5 | Sailboat         | 93606  | 97     |
| 6 | Kayak            | 46983  | 35     |
| 7 | Canoe            | 95026  | 32     |
| 8 | Houseboat        | 57685  | 100    |
| 9 | Pontoon          | 60323  | 43     |
|10 | Jet Ski          | 91224  | 15     |
|11 | Rowboat          | 44003  | 95     |
|12 | Hovercraft       | 75998  | 57     |
|13 | Cabin Cruiser    | 84525  | 13     |
|14 | Wakeboard Boat   | 66207  | 44     |
|15 | Dinghy           | 65002  | 64     |
|16 | Trawler          | 33132  | 88     |
|17 | Paddle Boat      | 69239  | 42     |
|18 | Submarine        | 66948  | 46     |
|19 | RIB              | 88240  | 24     |
|20 | Skiff            | 48858  | 93     |

Mathematical Model:

Variables:
- For each i ∈ {1,...,14}, j ∈ {1,...,20}: x_{ij} ∈ {0, 1, 2, ...}

Objective:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given in products.csv.

Subject to (for each display area i, using DisplayID and Capacity from capacity.csv):

For each i ∈ {1,...,14}:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j is as given in products.csv, and Capacity_i is as given in capacity.csv.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,14\},\ j \in \{1,...,20\}
\]

Explicitly, for each display area i (DisplayID from capacity.csv):

For DisplayID = 1 (Capacity = 457):
\[
18x_{1,1} + 36x_{1,2} + 25x_{1,3} + 16x_{1,4} + 97x_{1,5} + 35x_{1,6} + 32x_{1,7} + 100x_{1,8} + 43x_{1,9} + 15x_{1,10} + 95x_{1,11} + 57x_{1,12} + 13x_{1,13} + 44x_{1,14} + 64x_{1,15} + 88x_{1,16} + 42x_{1,17} + 46x_{1,18} + 24x_{1,19} + 93x_{1,20} \leq 457
\]
... and similarly for DisplayID = 2 to 14, using the corresponding Capacity.

Summary:
- Maximize total value of all vessels placed in all display areas.
- For each display area, the total weight of vessels placed cannot exceed its capacity.
- Decision variables are nonnegative integers: number of each vessel type in each display area.

This is a multiple knapsack integer programming problem with 14 knapsacks (display areas) and 20 item types (vessel types), with all data and indices as above.