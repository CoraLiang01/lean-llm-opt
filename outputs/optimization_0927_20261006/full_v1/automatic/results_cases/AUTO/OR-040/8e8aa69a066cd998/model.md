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

- Benefit coefficients $p_i$ and development unit weights $w_i$ for each area $i$:

| Area                | $p_i$ (Value) | $w_i$ (Weight) |
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

- Total development capacity: $C = 4466$

Mathematical Model:

Objective:
$$
\max \left(
443x_1 + 522x_2 + 300x_3 + 767x_4 + 300x_5 + 309x_6 + 598x_7 + 460x_8 + 318x_9 + 126x_{10} + 593x_{11} + 871x_{12} + 858x_{13} + 321x_{14} + 275x_{15} + 700x_{16} + 685x_{17} + 940x_{18} + 522x_{19} + 763x_{20}
\right)
$$

Subject to:
$$
104x_1 + 368x_2 + 483x_3 + 165x_4 + 105x_5 + 123x_6 + 131x_7 + 341x_8 + 258x_9 + 469x_{10} + 387x_{11} + 425x_{12} + 482x_{13} + 495x_{14} + 305x_{15} + 377x_{16} + 318x_{17} + 56x_{18} + 213x_{19} + 472x_{20} \leq 4466
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
$$

Where $x_i$ is the integer number of development units allocated daily to area $i$ as listed above.