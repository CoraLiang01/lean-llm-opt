Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars.

Parameters:
- $v_i$: Value (development benefit) for area $i$
- $w_i$: Weight (resource consumption per unit scale) for area $i$
- $C$: Overall development capacity

Data (in source order):

Capacity:
- $C = 586$

Products:

| Area (ProductName)      | $v_i$ (Value) | $w_i$ (Weight) |
|------------------------|--------------|---------------|
| Queens                 | 469          | 954           |
| Brooklyn               | 290          | 650           |
| Manhattan              | 236          | 961           |
| Bronx                  | 235          | 950           |
| Staten Island          | 745          | 379           |
| Harlem                 | 684          | 776           |
| Upper East Side        | 444          | 381           |
| Lower Manhattan        | 172          | 808           |
| Midtown                | 1000         | 937           |
| Long Island City       | 336          | 608           |
| Williamsburg           | 546          | 912           |
| Bushwick               | 535          | 391           |
| Flatbush               | 539          | 465           |
| Greenpoint             | 831          | 490           |
| Park Slope             | 139          | 918           |
| Astoria                | 432          | 787           |
| Jackson Heights        | 627          | 347           |
| Flushing               | 629          | 274           |
| Sunnyside              | 292          | 642           |
| Ditmars                | 978          | 130           |

Mathematical Model:

Objective:
$$
\max \sum_{i} v_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 586
$$

$$
x_i \geq 0 \quad \text{and integer}, \quad \forall i \in \{\text{areas listed above}\}
$$

Where:
- $v_i$, $w_i$ as given in the table above for each area $i$
- $x_i$ is the scale of development per day in area $i$ (nonnegative integer)