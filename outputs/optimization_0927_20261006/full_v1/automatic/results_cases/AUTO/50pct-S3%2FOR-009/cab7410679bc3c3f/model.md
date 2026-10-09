Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas in the order given:

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

The parameters for each area $i$ are:

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

The overall development capacity is:

- Capacity: $586$

The mathematical model is:

$$
\begin{align*}
\max \quad & 469x_1 + 290x_2 + 236x_3 + 235x_4 + 745x_5 + 684x_6 + 444x_7 + 172x_8 + 1000x_9 + 336x_{10} \\
& + 546x_{11} + 535x_{12} + 539x_{13} + 831x_{14} + 139x_{15} + 432x_{16} + 627x_{17} + 629x_{18} + 292x_{19} + 978x_{20} \\
\text{s.t.} \quad & 954x_1 + 650x_2 + 961x_3 + 950x_4 + 379x_5 + 776x_6 + 381x_7 + 808x_8 + 937x_9 + 608x_{10} \\
& + 912x_{11} + 391x_{12} + 465x_{13} + 490x_{14} + 918x_{15} + 787x_{16} + 347x_{17} + 274x_{18} + 642x_{19} + 130x_{20} \leq 586 \\
& x_i \geq 0, \quad \forall i = 1, \ldots, 20
\end{align*}
$$

Where:

- $x_i$ is the scale of development per day in area $i$ (continuous, nonnegative).
- $v_i$ is the benefit per unit development in area $i$ (from "Value" column).
- $w_i$ is the resource consumption per unit development in area $i$ (from "Weight" column).
- The total weighted development cannot exceed the overall capacity of $586$.

All coefficients and identifiers are as retrieved, in original file order.