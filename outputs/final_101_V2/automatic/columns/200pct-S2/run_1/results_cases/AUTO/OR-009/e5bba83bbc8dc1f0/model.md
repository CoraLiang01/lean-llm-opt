Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed by ProductName in products.csv.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value for area $i$ from products.csv.

Subject to:
$$
\sum_{i} w_i x_i \leq 586
$$
where $w_i$ is the Weight for area $i$ from products.csv, and 586 is the overall development capacity from capacity.csv.

$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i
$$

Where:

| $i$ (ProductName)      | $v_i$ (Value) | $w_i$ (Weight) |
|-----------------------|--------------|---------------|
| Queens                | 469          | 954           |
| Brooklyn              | 290          | 650           |
| Manhattan             | 236          | 961           |
| Bronx                 | 235          | 950           |
| Staten Island         | 745          | 379           |
| Harlem                | 684          | 776           |
| Upper East Side       | 444          | 381           |
| Lower Manhattan       | 172          | 808           |
| Midtown               | 1000         | 937           |
| Long Island City      | 336          | 608           |
| Williamsburg          | 546          | 912           |
| Bushwick              | 535          | 391           |
| Flatbush              | 539          | 465           |
| Greenpoint            | 831          | 490           |
| Park Slope            | 139          | 918           |
| Astoria               | 432          | 787           |
| Jackson Heights       | 627          | 347           |
| Flushing              | 629          | 274           |
| Sunnyside             | 292          | 642           |
| Ditmars               | 978          | 130           |

All variables $x_i$ are nonnegative integers.