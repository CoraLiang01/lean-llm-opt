Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas:

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

Let $v_i$ be the "Value" for area $i$ (from products.csv), and $w_i$ be the "Weight" for area $i$ (from products.csv). Let $C$ be the overall development capacity (from capacity.csv).

Numerical values:

- Capacity: $C = 586$
- For each area $i$:

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

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & \sum_{i} v_i x_i \\
\text{subject to} \quad & \sum_{i} w_i x_i \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

Where:

- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value for area $i$ (see table above)
- $w_i$ = Weight for area $i$ (see table above)
- $586$ = overall development capacity

All areas and their coefficients are included as retrieved, in original order.