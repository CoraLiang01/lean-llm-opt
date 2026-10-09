#### Sets and Indices

Let $I$ be the set of areas (indexed by $i$), with the following elements in source order:
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

#### Parameters

For each area $i$:
- $v_i$ = Value (development benefit per unit scale)
- $w_i$ = Weight (resource consumption per unit scale)

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

Total development capacity: $C = 586$

#### Decision Variables

For each area $i$:
- $x_i \geq 0$ : scale of development per day in area $i$ (continuous, as the question does not specify integrality)

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq 586
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

#### Parameter Table (source order)

| Area                | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Queens              | 469          | 954           |
| Brooklyn            | 290          | 650           |
| Manhattan           | 236          | 961           |
| Bronx               | 235          | 950           |
| Staten Island       | 745          | 379           |
| Harlem              | 684          | 776           |
| Upper East Side     | 444          | 381           |
| Lower Manhattan     | 172          | 808           |
| Midtown             | 1000         | 937           |
| Long Island City    | 336          | 608           |
| Williamsburg        | 546          | 912           |
| Bushwick            | 535          | 391           |
| Flatbush            | 539          | 465           |
| Greenpoint          | 831          | 490           |
| Park Slope          | 139          | 918           |
| Astoria             | 432          | 787           |
| Jackson Heights     | 627          | 347           |
| Flushing            | 629          | 274           |
| Sunnyside           | 292          | 642           |
| Ditmars             | 978          | 130           |

**Total Capacity:** $586$