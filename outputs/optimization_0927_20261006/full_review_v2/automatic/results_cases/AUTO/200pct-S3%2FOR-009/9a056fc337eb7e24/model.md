Let $x_i$ be the scale of development per day in area $i$, for each area $i$ listed below. All $x_i$ are nonnegative integers.

**Parameters:**

- For each area $i$:
    - $v_i$ = Value (development benefit per unit)
    - $w_i$ = Weight (resource requirement per unit)
- $C$ = overall development capacity

**Data:**

- Overall development capacity: $C = 586$
- Areas and coefficients:

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

**Mathematical Model:**

Objective:
$$
\max \sum_{i} v_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where $i$ ranges over the following areas (in source order):

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

With the corresponding $v_i$ and $w_i$ as listed above.