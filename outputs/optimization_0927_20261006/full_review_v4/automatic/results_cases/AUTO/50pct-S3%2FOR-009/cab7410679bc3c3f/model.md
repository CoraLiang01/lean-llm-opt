Let $I$ be the set of areas (indexed by $i$), with data as below.

Define decision variables:
$$
x_i = \text{scale of development per day in area } i, \quad x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i
$$

Parameters (from the retrieved data):

| Area (ProductName)      | Value ($v_i$) | Weight ($w_i$) |
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
| Astoria                | 432          | 787           |
| Jackson Heights        | 627          | 347           |
| Flushing               | 629          | 274           |
| Sunnyside              | 292          | 642           |
| Ditmars                | 978          | 130           |

Total development capacity (from capacity.csv): $C = 586$

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

Where:
- $v_i$ is the Value for area $i$ (from products.csv)
- $w_i$ is the Weight for area $i$ (from products.csv)
- $C = 586$ is the overall development capacity (from capacity.csv)

All data is used in the order and with the identifiers as retrieved.