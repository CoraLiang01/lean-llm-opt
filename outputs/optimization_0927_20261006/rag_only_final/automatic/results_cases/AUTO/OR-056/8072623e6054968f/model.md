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
- Weight_j: size (weight) of vessel type j (from products.csv)

Variables:
- x_{ij} = number of vessels of type j assigned to display area i (integer, x_{ij} ≥ 0)

Data (in supplied order):

Display area capacities (capacity.csv):

| DisplayID (i) | Capacity_i |
|---------------|------------|
| 1             | 457        |
| 2             | 604        |
| 3             | 751        |
| 4             | 468        |
| 5             | 343        |
| 6             | 408        |
| 7             | 741        |
| 8             | 914        |
| 9             | 682        |
| 10            | 409        |
| 11            | 342        |
| 12            | 903        |
| 13            | 680        |
| 14            | 886        |

Vessel types (products.csv):

| j  | ProductName      | Value_j | Weight_j |
|----|------------------|---------|----------|
| 1  | Speedboat        | 29664   | 18       |
| 2  | Fishing Boat     | 31778   | 36       |
| 3  | Catamaran        | 73501   | 25       |
| 4  | Yacht            | 78255   | 16       |
| 5  | Sailboat         | 93606   | 97       |
| 6  | Kayak            | 46983   | 35       |
| 7  | Canoe            | 95026   | 32       |
| 8  | Houseboat        | 57685   | 100      |
| 9  | Pontoon          | 60323   | 43       |
| 10 | Jet Ski          | 91224   | 15       |
| 11 | Rowboat          | 44003   | 95       |
| 12 | Hovercraft       | 75998   | 57       |
| 13 | Cabin Cruiser    | 84525   | 13       |
| 14 | Wakeboard Boat   | 66207   | 44       |
| 15 | Dinghy           | 65002   | 64       |
| 16 | Trawler          | 33132   | 88       |
| 17 | Paddle Boat      | 69239   | 42       |
| 18 | Submarine        | 66948   | 46       |
| 19 | RIB              | 88240   | 24       |
| 20 | Skiff            | 48858   | 93       |

Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

Subject to capacity constraints for each display area:
\[
\forall i \in \{1,\ldots,14\}: \quad \sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
\]

Where all coefficients and identifiers are as given in the tables above, preserving the supplied file and row order.