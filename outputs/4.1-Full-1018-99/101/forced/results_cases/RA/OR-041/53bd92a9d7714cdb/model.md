Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed below. All $x_i$ are nonnegative integers.

**Parameters:**

- For each area $i$:
    - $v_i$ = Value (development benefit per unit)
    - $w_i$ = Weight (resource consumption per unit)

- Total development capacity: $C = 586$

**Areas, Values, and Weights (in source order):**

| Area                | $v_i$ | $w_i$ |
|---------------------|-------|-------|
| Queens              | 469   | 954   |
| Brooklyn            | 290   | 650   |
| Manhattan           | 236   | 961   |
| Bronx               | 235   | 950   |
| Staten Island       | 745   | 379   |
| Harlem              | 684   | 776   |
| Upper East Side     | 444   | 381   |
| Lower Manhattan     | 172   | 808   |
| Midtown             | 1000  | 937   |
| Long Island City    | 336   | 608   |
| Williamsburg        | 546   | 912   |
| Bushwick            | 535   | 391   |
| Flatbush            | 539   | 465   |
| Greenpoint          | 831   | 490   |
| Park Slope          | 139   | 918   |
| Astoria             | 432   | 787   |
| Jackson Heights     | 627   | 347   |
| Flushing            | 629   | 274   |
| Sunnyside           | 292   | 642   |
| Ditmars             | 978   | 130   |

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i} v_i x_i
$$

Subject to the overall development capacity:
$$
\sum_{i} w_i x_i \leq 586
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where the indices $i$ and their corresponding $v_i$ and $w_i$ are as listed above.