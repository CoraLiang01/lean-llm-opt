Let:
- \( I \) be the set of display areas, indexed by \( i \), with DisplayIDs from capacity.csv.
- \( J \) be the set of boat types, indexed by \( j \), with ProductNames from products.csv.
- \( C_i \) be the capacity of display area \( i \) (from the "Capacity" column in capacity.csv).
- \( v_j \) be the value of boat type \( j \) (from the "Value" column in products.csv).
- \( w_j \) be the size (weight) of boat type \( j \) (from the "Weight" column in products.csv).
- \( x_{ij} \) be the integer decision variable: number of units of boat type \( j \) to place in display area \( i \).

The model is:

Sets and Data:
- Display areas \( I = \{1,2,\ldots,14\} \) with capacities:
    - \( C_1 = 356 \)
    - \( C_2 = 478 \)
    - \( C_3 = 305 \)
    - \( C_4 = 291 \)
    - \( C_5 = 168 \)
    - \( C_6 = 449 \)
    - \( C_7 = 139 \)
    - \( C_8 = 383 \)
    - \( C_9 = 472 \)
    - \( C_{10} = 288 \)
    - \( C_{11} = 320 \)
    - \( C_{12} = 250 \)
    - \( C_{13} = 402 \)
    - \( C_{14} = 293 \)
- Boat types \( J \) with (ProductName, Value, Weight):

| \( j \) | ProductName      | \( v_j \) | \( w_j \) |
|---------|------------------|-----------|-----------|
| 1       | Speedboat        | 69978     | 18        |
| 2       | Fishing Boat     | 54011     | 42        |
| 3       | Catamaran        | 36352     | 49        |
| 4       | Yacht            | 51521     | 42        |
| 5       | Sailboat         | 50415     | 41        |
| 6       | Kayak            | 76109     | 48        |
| 7       | Canoe            | 50462     | 22        |
| 8       | Houseboat        | 28989     | 29        |
| 9       | Pontoon          | 23318     | 45        |
| 10      | Jet Ski          | 26142     | 14        |
| 11      | Rowboat          | 42040     | 38        |
| 12      | Hovercraft       | 85961     | 47        |
| 13      | Cabin Cruiser    | 50142     | 45        |
| 14      | Wakeboard Boat   | 48478     | 28        |
| 15      | Dinghy           | 60953     | 24        |
| 16      | Trawler          | 95265     | 39        |
| 17      | Paddle Boat      | 22839     | 32        |
| 18      | Submarine        | 90957     | 36        |
| 19      | RIB              | 84652     | 14        |
| 20      | Skiff            | 78991     | 16        |

Decision variables:
- For each \( i \in I \), \( j \in J \):
    - \( x_{ij} \in \mathbb{Z}_{\geq 0} \) (nonnegative integers)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to (for each display area \( i \)):
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Where all coefficients and sets are as listed above, and the indices correspond to the explicit DisplayID and ProductName as given in the source data.