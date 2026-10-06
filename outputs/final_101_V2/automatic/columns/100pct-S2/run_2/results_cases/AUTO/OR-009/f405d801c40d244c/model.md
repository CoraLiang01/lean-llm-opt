Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed by ProductName in products.csv.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value for area $i$.

Subject to:

Capacity constraint:
$$
\sum_{i} w_i x_i \leq 586
$$
where $w_i$ is the Weight for area $i$ (from products.csv), and 586 is the Capacity from capacity.csv.

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Queens              | 469   | 954    |
| Brooklyn            | 290   | 650    |
| Manhattan           | 236   | 961    |
| Bronx               | 235   | 950    |
| Staten Island       | 745   | 379    |
| Harlem              | 684   | 776    |
| Upper East Side     | 444   | 381    |
| Lower Manhattan     | 172   | 808    |
| Midtown             | 1000  | 937    |
| Long Island City    | 336   | 608    |
| Williamsburg        | 546   | 912    |
| Bushwick            | 535   | 391    |
| Flatbush            | 539   | 465    |
| Greenpoint          | 831   | 490    |
| Park Slope          | 139   | 918    |
| Astoria             | 432   | 787    |
| Jackson Heights     | 627   | 347    |
| Flushing            | 629   | 274    |
| Sunnyside           | 292   | 642    |
| Ditmars             | 978   | 130    |

All $x_i$ are nonnegative integers. The objective is to maximize total Value, subject to the total Weight not exceeding 586.