##### Decision Variables

Let $x_i \geq 0$ be the scale of development per day in area $i$, for each area $i$ listed below.

##### Parameters

- $v_i$: development benefit per unit in area $i$
- $w_i$: development resource usage per unit in area $i$
- $C$: overall development capacity

Areas, benefits, and weights (from products.csv, in source order):

| Area                | $v_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Queens              | 469           | 954            |
| Brooklyn            | 290           | 650            |
| Manhattan           | 236           | 961            |
| Bronx               | 235           | 950            |
| Staten Island       | 745           | 379            |
| Harlem              | 684           | 776            |
| Upper East Side     | 444           | 381            |
| Lower Manhattan     | 172           | 808            |
| Midtown             | 1000          | 937            |
| Long Island City    | 336           | 608            |
| Williamsburg        | 546           | 912            |
| Bushwick            | 535           | 391            |
| Flatbush            | 539           | 465            |
| Greenpoint          | 831           | 490            |
| Park Slope          | 139           | 918            |
| Astoria             | 432           | 787            |
| Jackson Heights     | 627           | 347            |
| Flushing            | 629           | 274            |
| Sunnyside           | 292           | 642            |
| Ditmars             | 978           | 130            |

Overall development capacity (from capacity.csv):

$C = 586$

##### Mathematical Model

$\max \sum_{i} v_i x_i$

subject to

$\sum_{i} w_i x_i \leq 586$

$x_i \geq 0$ for all areas $i$ listed above.

Where the indices $i$ run over the areas in the table, with their respective $v_i$ and $w_i$ coefficients as shown.