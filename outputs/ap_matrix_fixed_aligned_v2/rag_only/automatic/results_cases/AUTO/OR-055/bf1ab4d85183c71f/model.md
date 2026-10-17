Let:
- \( I = \{1, 2, \ldots, 14\} \) be the set of display areas, indexed by DisplayID as given in capacity.csv.
- \( J = \{\text{Speedboat}, \text{Fishing Boat}, \text{Catamaran}, \text{Yacht}, \text{Sailboat}, \text{Kayak}, \text{Canoe}, \text{Houseboat}, \text{Pontoon}, \text{Jet Ski}, \text{Rowboat}, \text{Hovercraft}, \text{Cabin Cruiser}, \text{Wakeboard Boat}, \text{Dinghy}, \text{Trawler}, \text{Paddle Boat}, \text{Submarine}, \text{RIB}, \text{Skiff}\} \) be the set of boat types, indexed by ProductName as given in products.csv.

Parameters:
- \( C_i \): Capacity of display area \( i \) (from capacity.csv)
- \( v_j \): Value of boat type \( j \) (from products.csv)
- \( w_j \): Size (Weight) of boat type \( j \) (from products.csv)

Decision variables:
- \( x_{ij} \): Number of units of boat type \( j \) to place in display area \( i \), integer, \( x_{ij} \geq 0 \)

Data:
Display area capacities (from capacity.csv, in original order):

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

Boat types, values, and sizes (from products.csv, in original order):

| ProductName      | Value  | Weight |
|------------------|--------|--------|
| Speedboat        | 69978  | 18     |
| Fishing Boat     | 54011  | 42     |
| Catamaran        | 36352  | 49     |
| Yacht            | 51521  | 42     |
| Sailboat         | 50415  | 41     |
| Kayak            | 76109  | 48     |
| Canoe            | 50462  | 22     |
| Houseboat        | 28989  | 29     |
| Pontoon          | 23318  | 45     |
| Jet Ski          | 26142  | 14     |
| Rowboat          | 42040  | 38     |
| Hovercraft       | 85961  | 47     |
| Cabin Cruiser    | 50142  | 45     |
| Wakeboard Boat   | 48478  | 28     |
| Dinghy           | 60953  | 24     |
| Trawler          | 95265  | 39     |
| Paddle Boat      | 22839  | 32     |
| Submarine        | 90957  | 36     |
| RIB              | 84652  | 14     |
| Skiff            | 78991  | 16     |

Model:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i \in I} \sum_{j \in J} v_j x_{ij} \\
\text{subject to} \quad
& \sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, \forall j \in J
\end{align*}
\]

Where:
- \( v_j \) and \( w_j \) are as given in the table above for each ProductName \( j \).
- \( C_i \) is as given in the table above for each DisplayID \( i \).
- All variables \( x_{ij} \) are nonnegative integers.

This model maximizes the total value of boats displayed, subject to the size (weight) of boats in each display area not exceeding its capacity. Each \( x_{ij} \) is the integer number of units of boat type \( j \) in display area \( i \). All data is used in the original supplied order.