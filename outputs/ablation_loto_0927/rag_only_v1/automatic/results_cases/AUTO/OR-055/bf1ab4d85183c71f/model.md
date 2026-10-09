Let:
- \( x_{ij} \): integer number of units of boat type \( j \) (ProductName from products.csv) to be placed in display area \( i \) (DisplayID from capacity.csv), for all \( i = 1,\ldots,14 \), \( j = 1,\ldots,20 \).

Parameters:
- \( v_j \): Value of boat type \( j \) (from products.csv, column Value)
- \( w_j \): Weight (size) of boat type \( j \) (from products.csv, column Weight)
- \( C_i \): Capacity of display area \( i \) (from capacity.csv, column Capacity)

Indices:
- \( i \in \{1,2,\ldots,14\} \) (DisplayID from capacity.csv, in order)
- \( j \in \) {Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff} (ProductName from products.csv, in order)

Model:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij} \\
\text{Subject to:} \quad
& \sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,14,\ j = 1,\ldots,20
\end{align*}
\]

Where the data is:

Display areas (from capacity.csv, in order):

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

Boat types (from products.csv, in order):

| j | ProductName      | Value  | Weight |
|---|------------------|--------|--------|
| 1 | Speedboat        | 69978  | 18     |
| 2 | Fishing Boat     | 54011  | 42     |
| 3 | Catamaran        | 36352  | 49     |
| 4 | Yacht            | 51521  | 42     |
| 5 | Sailboat         | 50415  | 41     |
| 6 | Kayak            | 76109  | 48     |
| 7 | Canoe            | 50462  | 22     |
| 8 | Houseboat        | 28989  | 29     |
| 9 | Pontoon          | 23318  | 45     |
|10 | Jet Ski          | 26142  | 14     |
|11 | Rowboat          | 42040  | 38     |
|12 | Hovercraft       | 85961  | 47     |
|13 | Cabin Cruiser    | 50142  | 45     |
|14 | Wakeboard Boat   | 48478  | 28     |
|15 | Dinghy           | 60953  | 24     |
|16 | Trawler          | 95265  | 39     |
|17 | Paddle Boat      | 22839  | 32     |
|18 | Submarine        | 90957  | 36     |
|19 | RIB              | 84652  | 14     |
|20 | Skiff            | 78991  | 16     |

Explicitly, the model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{14} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij} \\
\text{Subject to:} \quad
& \sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,14,\ j = 1,\ldots,20
\end{align*}
\]

Where all coefficients and indices are as above, and all variables \( x_{ij} \) are nonnegative integers.