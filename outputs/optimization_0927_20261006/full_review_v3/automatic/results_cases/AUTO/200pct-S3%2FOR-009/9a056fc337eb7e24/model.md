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

Let $v_i$ be the benefit per unit development in area $i$ (from the "Value" column), and $w_i$ be the resource requirement per unit development in area $i$ (from the "Weight" column). The overall development capacity is $586$.

The complete mathematical model is:

Objective:
$$
\max \; 469x_1 + 290x_2 + 236x_3 + 235x_4 + 745x_5 + 684x_6 + 444x_7 + 172x_8 + 1000x_9 + 336x_{10} + 546x_{11} + 535x_{12} + 539x_{13} + 831x_{14} + 139x_{15} + 432x_{16} + 627x_{17} + 629x_{18} + 292x_{19} + 978x_{20}
$$

Subject to:
$$
954x_1 + 650x_2 + 961x_3 + 950x_4 + 379x_5 + 776x_6 + 381x_7 + 808x_8 + 937x_9 + 608x_{10} + 912x_{11} + 391x_{12} + 465x_{13} + 490x_{14} + 918x_{15} + 787x_{16} + 347x_{17} + 274x_{18} + 642x_{19} + 130x_{20} \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 20
$$

Where:

- $x_i$ = scale of development per day in area $i$
- $v_i$ = benefit per unit development in area $i$ (see coefficients in the objective)
- $w_i$ = resource requirement per unit development in area $i$ (see coefficients in the constraint)
- The total resource used across all areas cannot exceed $586$.

All coefficients and area names are as retrieved and in the original order.