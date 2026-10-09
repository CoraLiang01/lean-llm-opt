Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

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

Parameters (from products.csv):

\[
\begin{array}{lll}
\text{Area (i)} & \text{Value } (v_i) & \text{Weight } (w_i) \\
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

Parameter (from capacity.csv):

- Overall development capacity: $C = 586$

Model:

Objective:
\[
\max \sum_{i} v_i x_i
\]

Subject to:
\[
\sum_{i} w_i x_i \leq 586
\]
\[
x_i \geq 0 \quad \text{and integer} \quad \forall i
\]

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value for area $i$ (see table above)
- $w_i$ = Weight for area $i$ (see table above)
- $C = 586$ (overall development capacity)

All identifiers and coefficients are as retrieved and in source order.