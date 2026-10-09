Let:
- \( I = \{1, 2, ..., 14\} \) index the display areas (DisplayID from capacity.csv, in order).
- \( J = \{1, 2, ..., 18\} \) index the boat types (ProductName from products.csv, in order).

Define:
- \( x_{ij} \): number of units of boat type \( j \) placed in display area \( i \), integer, \( x_{ij} \geq 0 \).

Parameters:
- For each display area \( i \) (DisplayID), capacity \( C_i \):

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

- For each boat type \( j \) (ProductName), value \( v_j \) and size \( s_j \):

| j | ProductName      | Value  | Size (Weight) |
|---|------------------|--------|--------------|
| 1 | Speedboat        | 69978  | 18           |
| 2 | Fishing Boat     | 54011  | 42           |
| 3 | Catamaran        | 36352  | 49           |
| 4 | Yacht            | 51521  | 42           |
| 5 | Sailboat         | 50415  | 41           |
| 6 | Kayak            | 76109  | 48           |
| 7 | Canoe            | 50462  | 22           |
| 8 | Houseboat        | 28989  | 29           |
| 9 | Pontoon          | 23318  | 45           |
|10 | Jet Ski          | 26142  | 14           |
|11 | Rowboat          | 42040  | 38           |
|12 | Hovercraft       | 85961  | 47           |
|13 | Cabin Cruiser    | 50142  | 45           |
|14 | Wakeboard Boat   | 48478  | 28           |
|15 | Dinghy           | 60953  | 24           |
|16 | Trawler          | 95265  | 39           |
|17 | Paddle Boat      | 22839  | 32           |
|18 | Submarine        | 90957  | 36           |
|19 | RIB              | 84652  | 14           |
|20 | Skiff            | 78991  | 16           |

(Note: There are actually 20 products in the data, not 18. So \( J = \{1, ..., 20\} \).)

Model:

Maximize total value:
\[
\max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j x_{ij}
\]
where \( v_j \) is the Value for ProductName \( j \) as above.

Subject to, for each display area \( i \) (DisplayID as above):
\[
\sum_{j=1}^{20} s_j x_{ij} \leq C_i
\]
where \( s_j \) is the Weight for ProductName \( j \), and \( C_i \) is the Capacity for DisplayID \( i \).

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1, ..., 14\},\ j \in \{1, ..., 20\}
\]

Where the mapping from \( j \) to ProductName, Value, and Weight is as listed above, and the mapping from \( i \) to DisplayID and Capacity is as listed above.

This is a complete integer programming formulation for the described problem, using all provided data and respecting all requirements.