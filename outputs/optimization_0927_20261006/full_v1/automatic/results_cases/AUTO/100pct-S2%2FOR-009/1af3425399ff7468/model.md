Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas in the order retrieved:

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

Let $v_i$ be the development benefit (Value) and $w_i$ be the development resource requirement (Weight) for area $i$ as given below:

\[
\begin{array}{lll}
\text{Area} & v_i & w_i \\
\hline
\text{Queens} & 469 & 954 \\
\text{Brooklyn} & 290 & 650 \\
\text{Manhattan} & 236 & 961 \\
\text{Bronx} & 235 & 950 \\
\text{Staten Island} & 745 & 379 \\
\text{Harlem} & 684 & 776 \\
\text{Upper East Side} & 444 & 381 \\
\text{Lower Manhattan} & 172 & 808 \\
\text{Midtown} & 1000 & 937 \\
\text{Long Island City} & 336 & 608 \\
\text{Williamsburg} & 546 & 912 \\
\text{Bushwick} & 535 & 391 \\
\text{Flatbush} & 539 & 465 \\
\text{Greenpoint} & 831 & 490 \\
\text{Park Slope} & 139 & 918 \\
\text{Astoria} & 432 & 787 \\
\text{Jackson Heights} & 627 & 347 \\
\text{Flushing} & 629 & 274 \\
\text{Sunnyside} & 292 & 642 \\
\text{Ditmars} & 978 & 130 \\
\end{array}
\]

The overall development capacity is $C = 586$.

The mathematical model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{20} v_i x_i \\
\text{subject to} \quad & \sum_{i=1}^{20} w_i x_i \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 20
\end{align*}
\]

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value for area $i$ (see table above)
- $w_i$ = Weight for area $i$ (see table above)
- $C = 586$ (overall development capacity)

All coefficients and area names are as retrieved and in the original order.