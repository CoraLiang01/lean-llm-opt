Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (in source order):

\[
\begin{array}{ll}
\text{Queens} & \text{Brooklyn} \\
\text{Manhattan} & \text{Bronx} \\
\text{Staten Island} & \text{Harlem} \\
\text{Upper East Side} & \text{Lower Manhattan} \\
\text{Midtown} & \text{Long Island City} \\
\text{Williamsburg} & \text{Bushwick} \\
\text{Flatbush} & \text{Greenpoint} \\
\text{Park Slope} & \text{Astoria} \\
\text{Jackson Heights} & \text{Flushing} \\
\text{Sunnyside} & \text{Ditmars} \\
\end{array}
\]

The development benefit (Value) and development capacity required (Weight) for each area are as follows (in source order):

\[
\begin{array}{lll}
\text{Area} & \text{Value}_i & \text{Weight}_i \\
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

The overall development capacity is $586$.

The mathematical model is:

\[
\textbf{Objective:} \quad \max \sum_{i} \text{Value}_i \cdot x_i
\]

\[
\textbf{Subject to:}
\]
\[
\sum_{i} \text{Weight}_i \cdot x_i \leq 586
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $\text{Value}_i$ = development benefit for area $i$ (see table above)
- $\text{Weight}_i$ = development capacity required for area $i$ (see table above)

All data and identifiers are preserved in source order.