Let:
- i index the display areas, with DisplayID from capacity.csv (i ∈ {1,2,...,14})
- j index the boat types, with ProductName from products.csv (j ∈ {Speedboat, Fishing Boat, ..., Skiff})
- x_{ij} = number of units of boat type j placed in display area i (integer, x_{ij} ≥ 0)

Parameters:
From capacity.csv:
- Capacity_i = capacity of display area i

From products.csv:
- Value_j = value of boat type j
- Weight_j = size of boat type j

Data:
Display areas (capacity.csv):

| DisplayID | Capacity |
|-----------|----------|
| 1         | 356      |
| 2         | 478      |
| 3         | 305      |
| 4         | 291      |
| 5         | 168      |
| 6         | 449      |
| 7         | 139      |
| 8         | 383      |
| 9         | 472      |
| 10        | 288      |
| 11        | 320      |
| 12        | 250      |
| 13        | 402      |
| 14        | 293      |

Boat types (products.csv):

| j               | Value_j | Weight_j |
|-----------------|---------|----------|
| Speedboat       | 69978   | 18       |
| Fishing Boat    | 54011   | 42       |
| Catamaran       | 36352   | 49       |
| Yacht           | 51521   | 42       |
| Sailboat        | 50415   | 41       |
| Kayak           | 76109   | 48       |
| Canoe           | 50462   | 22       |
| Houseboat       | 28989   | 29       |
| Pontoon         | 23318   | 45       |
| Jet Ski         | 26142   | 14       |
| Rowboat         | 42040   | 38       |
| Hovercraft      | 85961   | 47       |
| Cabin Cruiser   | 50142   | 45       |
| Wakeboard Boat  | 48478   | 28       |
| Dinghy          | 60953   | 24       |
| Trawler         | 95265   | 39       |
| Paddle Boat     | 22839   | 32       |
| Submarine       | 90957   | 36       |
| RIB             | 84652   | 14       |
| Skiff           | 78991   | 16       |

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,14}, j ∈ {1,...,20}

Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

Subject to, for each display area i (DisplayID from 1 to 14):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,\ldots,14\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,14\},\ j \in \{1,\ldots,20\}
\]

Where:
- Value_j and Weight_j are as given in products.csv for each ProductName j, in the original row order.
- Capacity_i is as given in capacity.csv for each DisplayID i, in the original row order.

This is a complete integer programming formulation for the described allocation problem, using all provided data and respecting all requirements.