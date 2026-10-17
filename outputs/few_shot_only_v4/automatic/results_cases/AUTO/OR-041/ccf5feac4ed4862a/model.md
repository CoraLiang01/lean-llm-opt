Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed in the table below.

**Parameters:**

- $v_i$: Value (development benefit) for area $i$
- $w_i$: Weight (resource consumption per unit development) for area $i$
- $C$: Overall development capacity

**Data:**

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

Overall development capacity: $C = 586$

**Mathematical Model:**

Objective:
\[
\max \sum_{i} v_i x_i
\]

Subject to:
\[
\sum_{i} w_i x_i \leq 586
\]
\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i
\]

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value for area $i$ (see table)
- $w_i$ = Weight for area $i$ (see table)
- $C = 586$ (overall development capacity)