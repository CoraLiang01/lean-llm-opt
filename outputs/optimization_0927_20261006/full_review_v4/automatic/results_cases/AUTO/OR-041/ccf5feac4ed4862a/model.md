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

Let $v_i$ be the development benefit (Value) and $w_i$ be the development resource requirement (Weight) for area $i$.

The overall development capacity is $586$.

The mathematical model is:

Objective:
\[
\max \;
469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}}
\]

Subject to:

\[
954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

\[
\begin{align*}
&x_{\text{Queens}}: \text{scale of development per day in Queens} \\
&x_{\text{Brooklyn}}: \text{scale of development per day in Brooklyn} \\
&x_{\text{Manhattan}}: \text{scale of development per day in Manhattan} \\
&x_{\text{Bronx}}: \text{scale of development per day in Bronx} \\
&x_{\text{Staten Island}}: \text{scale of development per day in Staten Island} \\
&x_{\text{Harlem}}: \text{scale of development per day in Harlem} \\
&x_{\text{Upper East Side}}: \text{scale of development per day in Upper East Side} \\
&x_{\text{Lower Manhattan}}: \text{scale of development per day in Lower Manhattan} \\
&x_{\text{Midtown}}: \text{scale of development per day in Midtown} \\
&x_{\text{Long Island City}}: \text{scale of development per day in Long Island City} \\
&x_{\text{Williamsburg}}: \text{scale of development per day in Williamsburg} \\
&x_{\text{Bushwick}}: \text{scale of development per day in Bushwick} \\
&x_{\text{Flatbush}}: \text{scale of development per day in Flatbush} \\
&x_{\text{Greenpoint}}: \text{scale of development per day in Greenpoint} \\
&x_{\text{Park Slope}}: \text{scale of development per day in Park Slope} \\
&x_{\text{Astoria}}: \text{scale of development per day in Astoria} \\
&x_{\text{Jackson Heights}}: \text{scale of development per day in Jackson Heights} \\
&x_{\text{Flushing}}: \text{scale of development per day in Flushing} \\
&x_{\text{Sunnyside}}: \text{scale of development per day in Sunnyside} \\
&x_{\text{Ditmars}}: \text{scale of development per day in Ditmars} \\
\end{align*}
\]