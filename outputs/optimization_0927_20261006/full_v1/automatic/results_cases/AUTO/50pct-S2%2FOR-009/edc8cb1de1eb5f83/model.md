Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

- Queens
- Brooklyn
- Manhattan
- Bronx
- Staten Island
- Harlem
- Upper East Side
- Lower Manhattan
- Midtown
- Long Island City
- Williamsburg
- Bushwick
- Flatbush
- Greenpoint
- Park Slope
- Astoria
- Jackson Heights
- Flushing
- Sunnyside
- Ditmars

Let $v_i$ be the Value (development benefit) and $w_i$ be the Weight (resource requirement) for area $i$ as given below:

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

The overall development capacity is:

- Capacity = 586

The mathematical model is:

$$
\begin{align*}
\max \quad & \sum_{i} v_i x_i \\
\text{s.t.} \quad & \sum_{i} w_i x_i \leq 586 \\
& x_i \geq 0, \quad \forall i
\end{align*}
$$

Where:
- $x_i$ is the scale of development per day in area $i$ (continuous, nonnegative).
- $v_i$ and $w_i$ are as given in the table above.
- The sum of resource requirements across all selected areas cannot exceed 586.

All coefficients and identifiers are as retrieved and in original order.