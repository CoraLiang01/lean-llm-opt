Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed by ProductName in products.csv.

Parameters (from products.csv, in source order):

| $i$ | ProductName           | Value ($b_i$) | Weight ($a_i$) |
|-----|----------------------|--------------|---------------|
| 1   | Queens               | 469          | 954           |
| 2   | Brooklyn             | 290          | 650           |
| 3   | Manhattan            | 236          | 961           |
| 4   | Bronx                | 235          | 950           |
| 5   | Staten Island        | 745          | 379           |
| 6   | Harlem               | 684          | 776           |
| 7   | Upper East Side      | 444          | 381           |
| 8   | Lower Manhattan      | 172          | 808           |
| 9   | Midtown              | 1000         | 937           |
| 10  | Long Island City     | 336          | 608           |
| 11  | Williamsburg         | 546          | 912           |
| 12  | Bushwick             | 535          | 391           |
| 13  | Flatbush             | 539          | 465           |
| 14  | Greenpoint           | 831          | 490           |
| 15  | Park Slope           | 139          | 918           |
| 16  | Astoria              | 432          | 787           |
| 17  | Jackson Heights      | 627          | 347           |
| 18  | Flushing             | 629          | 274           |
| 19  | Sunnyside            | 292          | 642           |
| 20  | Ditmars              | 978          | 130           |

Parameter (from capacity.csv):

- Overall development capacity: $C = 586$

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{20} b_i x_i
$$

Subject to:
$$
\sum_{i=1}^{20} a_i x_i \leq 586
$$

$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i = 1, \ldots, 20
$$

Where:
- $b_i$ is the Value for area $i$ (from products.csv)
- $a_i$ is the Weight for area $i$ (from products.csv)
- $x_i$ is the scale of development per day in area $i$ (decision variable, nonnegative integer)
- The total development "Weight" across all areas cannot exceed the overall capacity $C = 586$.

All coefficients and identifiers are as retrieved, in original file order.