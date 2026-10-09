Let $I$ be the set of areas, in source order:
$$
I = \{\text{Queens},\ \text{Brooklyn},\ \text{Manhattan},\ \text{Bronx},\ \text{Staten Island},\ \text{Harlem},\ \text{Upper East Side},\ \text{Lower Manhattan},\ \text{Midtown},\ \text{Long Island City},\ \text{Williamsburg},\ \text{Bushwick},\ \text{Flatbush},\ \text{Greenpoint},\ \text{Park Slope},\ \text{Astoria},\ \text{Jackson Heights},\ \text{Flushing},\ \text{Sunnyside},\ \text{Ditmars}\}
$$

Let $x_i$ be the integer scale of development in area $i$ per day.

Parameters (from source order):

\[
\begin{array}{lll}
\text{Area} & \text{Benefit Coefficient } (v_i) & \text{Unit Weight } (w_i) \\
\hline
\text{Queens} & 443 & 104 \\
\text{Brooklyn} & 522 & 368 \\
\text{Manhattan} & 300 & 483 \\
\text{Bronx} & 767 & 165 \\
\text{Staten Island} & 300 & 105 \\
\text{Harlem} & 309 & 123 \\
\text{Upper East Side} & 598 & 131 \\
\text{Lower Manhattan} & 460 & 341 \\
\text{Midtown} & 318 & 258 \\
\text{Long Island City} & 126 & 469 \\
\text{Williamsburg} & 593 & 387 \\
\text{Bushwick} & 871 & 425 \\
\text{Flatbush} & 858 & 482 \\
\text{Greenpoint} & 321 & 495 \\
\text{Park Slope} & 275 & 305 \\
\text{Astoria} & 700 & 377 \\
\text{Jackson Heights} & 685 & 318 \\
\text{Flushing} & 940 & 56 \\
\text{Sunnyside} & 522 & 213 \\
\text{Ditmars} & 763 & 472 \\
\end{array}
\]

Overall capacity: $C = 4466$

Mathematical Model:

Decision variables:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq 4466
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Where the parameters $v_i$ and $w_i$ are as listed above for each area $i$ in source order.