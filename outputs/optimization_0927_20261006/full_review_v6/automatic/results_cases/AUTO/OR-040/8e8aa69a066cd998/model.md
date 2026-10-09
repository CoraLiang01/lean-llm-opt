Let $x_i$ be the integer number of development units allocated daily to area $i$, where $i$ indexes the following areas in the order given:

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
15. Park Slope
16. Astoria
17. Jackson Heights
18. Flushing
19. Sunnyside
20. Ditmars

Parameters:

- $v_i$: Benefit coefficient for area $i$
- $w_i$: Development unit weight for area $i$
- $C$: Overall development capacity

Data:

| Area                | $v_i$ | $w_i$ |
|---------------------|-------|-------|
| Queens              | 443   | 104   |
| Brooklyn            | 522   | 368   |
| Manhattan           | 300   | 483   |
| Bronx               | 767   | 165   |
| Staten Island       | 300   | 105   |
| Harlem              | 309   | 123   |
| Upper East Side     | 598   | 131   |
| Lower Manhattan     | 460   | 341   |
| Midtown             | 318   | 258   |
| Long Island City    | 126   | 469   |
| Williamsburg        | 593   | 387   |
| Bushwick            | 871   | 425   |
| Flatbush            | 858   | 482   |
| Greenpoint          | 321   | 495   |
| Park Slope          | 275   | 305   |
| Astoria             | 700   | 377   |
| Jackson Heights     | 685   | 318   |
| Flushing            | 940   | 56    |
| Sunnyside           | 522   | 213   |
| Ditmars             | 763   | 472   |

Overall capacity: $C = 4466$

Mathematical Model:

$$
\max \sum_{i=1}^{20} v_i x_i
$$

subject to

$$
\sum_{i=1}^{20} w_i x_i \leq 4466
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
$$

where the $v_i$ and $w_i$ are as listed above for each area.