Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

1. Queens
2. Brooklyn
3. Manhattan
4. Bronx
5. Staten Island
6. Harlem
7. Upper East Side
8. Lower Manhattan
9. Midtown
10. Long Island City
11. Williamsburg
12. Bushwick
13. Flatbush
14. Greenpoint
15. Astoria
16. Jackson Heights
17. Flushing
18. Sunnyside
19. Ditmars

Parameters:

- $v_i$: Value (development benefit) for area $i$
- $w_i$: Weight (resource requirement) for area $i$
- $C$: Overall development capacity

From the data:

- $C = 586$
- $(v_i, w_i)$ for each area $i$ as follows (in source order):

| ProductName         | Value ($v_i$) | Weight ($w_i$) |
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
| Astoria             | 432          | 787           |
| Jackson Heights     | 627          | 347           |
| Flushing            | 629          | 274           |
| Sunnyside           | 292          | 642           |
| Ditmars             | 978          | 130           |

Model:

Objective:
\[
\max \sum_{i=1}^{19} v_i x_i
\]

Subject to:
\[
\sum_{i=1}^{19} w_i x_i \leq 586
\]
\[
x_i \geq 0 \quad \text{and integer}, \quad \forall i = 1, \ldots, 19
\]

Where:

- $v_i$ and $w_i$ are as listed above for each area $i$.
- $x_i$ is the scale of development per day in area $i$ (nonnegative integer).

Parameter Table (source order):

| $i$ | ProductName         | $v_i$ | $w_i$ |
|-----|---------------------|-------|-------|
| 1   | Queens              | 469   | 954   |
| 2   | Brooklyn            | 290   | 650   |
| 3   | Manhattan           | 236   | 961   |
| 4   | Bronx               | 235   | 950   |
| 5   | Staten Island       | 745   | 379   |
| 6   | Harlem              | 684   | 776   |
| 7   | Upper East Side     | 444   | 381   |
| 8   | Lower Manhattan     | 172   | 808   |
| 9   | Midtown             | 1000  | 937   |
| 10  | Long Island City    | 336   | 608   |
| 11  | Williamsburg        | 546   | 912   |
| 12  | Bushwick            | 535   | 391   |
| 13  | Flatbush            | 539   | 465   |
| 14  | Greenpoint          | 831   | 490   |
| 15  | Astoria             | 432   | 787   |
| 16  | Jackson Heights     | 627   | 347   |
| 17  | Flushing            | 629   | 274   |
| 18  | Sunnyside           | 292   | 642   |
| 19  | Ditmars             | 978   | 130   |

Capacity: $C = 586$

Decision variables: $x_i \in \mathbb{Z}_{\geq 0}$ for all $i = 1, \ldots, 19$.

Objective: Maximize total development benefit.  
Constraint: Total resource usage does not exceed 586.  
All variables are nonnegative integers.