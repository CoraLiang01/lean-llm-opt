Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed in the ProductName column.

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{Area (i)} & \text{Value } (p_i) & \text{Weight } (a_i) \\
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

\[
C = 586
\]

Model:

Maximize total benefit:
\[
\max \sum_{i} p_i x_i
\]

Subject to the overall development capacity:
\[
\sum_{i} a_i x_i \leq 586
\]

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
\]

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $p_i$ = Value for area $i$ (from products.csv)
- $a_i$ = Weight for area $i$ (from products.csv)
- $C$ = 586 (overall development capacity from capacity.csv)