Let:
- i index the display areas (DisplayID from capacity.csv, i = 1,...,14)
- j index the vessel types (ProductName from products.csv, j = 1,...,20)
- x_{ij} = number of vessels of type j placed in display area i (integer, x_{ij} ≥ 0)

Parameters:
- Capacity_i = capacity of display area i (from capacity.csv)
- Value_j = value of vessel type j (from products.csv)
- Weight_j = weight (size) of vessel type j (from products.csv)

Data:
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

Model:

Variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,14}, j ∈ {1,...,20}

Objective:
Maximize total value:
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
That is,
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} (\text{Value of ProductName}_j) \cdot x_{ij}
\]

Subject to, for each display area i (DisplayID from 1 to 14):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
That is, for each i:
\[
\sum_{j=1}^{20} (\text{Weight of ProductName}_j) \cdot x_{ij} \leq (\text{Capacity of DisplayID}_i)
\]

And for all i, j:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Where all coefficients and indices are as given in the tables above, preserving the original file and row order.

This is a multiple knapsack integer programming problem, maximizing the total value of boats assigned to display areas, subject to each area's capacity.