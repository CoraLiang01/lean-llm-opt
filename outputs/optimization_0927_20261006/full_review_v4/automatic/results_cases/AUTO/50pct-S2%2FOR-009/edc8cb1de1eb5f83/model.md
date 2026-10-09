Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (in source order):

\[
\begin{array}{ll}
1: \text{Queens} & 2: \text{Brooklyn} \\
3: \text{Manhattan} & 4: \text{Bronx} \\
5: \text{Staten Island} & 6: \text{Harlem} \\
7: \text{Upper East Side} & 8: \text{Lower Manhattan} \\
9: \text{Midtown} & 10: \text{Long Island City} \\
11: \text{Williamsburg} & 12: \text{Bushwick} \\
13: \text{Flatbush} & 14: \text{Greenpoint} \\
15: \text{Park Slope} & 16: \text{Astoria} \\
17: \text{Jackson Heights} & 18: \text{Flushing} \\
19: \text{Sunnyside} & 20: \text{Ditmars} \\
\end{array}
\]

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{Area} & \text{Value}~(v_i) & \text{Weight}~(w_i) \\
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

Total development capacity (from capacity.csv):

\[
\text{Total Capacity} = 586
\]

##### Mathematical Model

**Decision variables:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,20\}
\]

**Objective:**
\[
\max \sum_{i=1}^{20} v_i x_i
\]
where $v_i$ is the Value for area $i$ as listed above.

**Constraint:**
\[
\sum_{i=1}^{20} w_i x_i \leq 586
\]
where $w_i$ is the Weight for area $i$ as listed above.

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Explicitly:**

\[
\begin{align*}
\max\ & 469x_1 + 290x_2 + 236x_3 + 235x_4 + 745x_5 + 684x_6 + 444x_7 + 172x_8 + 1000x_9 + 336x_{10} \\
& + 546x_{11} + 535x_{12} + 539x_{13} + 831x_{14} + 139x_{15} + 432x_{16} + 627x_{17} + 629x_{18} + 292x_{19} + 978x_{20} \\
\text{s.t.}\ & 954x_1 + 650x_2 + 961x_3 + 950x_4 + 379x_5 + 776x_6 + 381x_7 + 808x_8 + 937x_9 + 608x_{10} \\
& + 912x_{11} + 391x_{12} + 465x_{13} + 490x_{14} + 918x_{15} + 787x_{16} + 347x_{17} + 274x_{18} + 642x_{19} + 130x_{20} \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,20\}
\end{align*}
\]