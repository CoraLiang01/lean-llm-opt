Let $x_i$ be the integer number of development units in area $i$ per day.

Maximize total benefit:
$$
\max \sum_{i} v_i x_i
$$

Subject to the overall development capacity:
$$
\sum_{i} w_i x_i \leq 4466
$$

and integer, nonnegative variables:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

where the areas $i$ and their coefficients are:

| ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Queens              | 443           | 104            |
| Brooklyn            | 522           | 368            |
| Manhattan           | 300           | 483            |
| Bronx               | 767           | 165            |
| Staten Island       | 300           | 105            |
| Harlem              | 309           | 123            |
| Upper East Side     | 598           | 131            |
| Lower Manhattan     | 460           | 341            |
| Midtown             | 318           | 258            |
| Long Island City    | 126           | 469            |
| Williamsburg        | 593           | 387            |
| Bushwick            | 871           | 425            |
| Flatbush            | 858           | 482            |
| Greenpoint          | 321           | 495            |
| Park Slope          | 275           | 305            |
| Astoria             | 700           | 377            |
| Jackson Heights     | 685           | 318            |
| Flushing            | 940           | 56             |
| Sunnyside           | 522           | 213            |
| Ditmars             | 763           | 472            |

All variables $x_i$ are integer and nonnegative. The objective is to maximize total benefit, subject to the total development capacity of 4466 units.