Let $I$ be the set of areas (indexed by ProductName in source order):

\[
I = \{\text{Queens},\ \text{Brooklyn},\ \text{Manhattan},\ \text{Bronx},\ \text{Staten Island},\ \text{Harlem},\ \text{Upper East Side},\ \text{Lower Manhattan},\ \text{Midtown},\ \text{Long Island City},\ \text{Williamsburg},\ \text{Bushwick},\ \text{Flatbush},\ \text{Greenpoint},\ \text{Astoria},\ \text{Jackson Heights},\ \text{Flushing},\ \text{Sunnyside},\ \text{Ditmars}\}
\]

Let $x_i$ be the scale of development per day in area $i$, for each $i \in I$.

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{Area} & \text{Value } (b_i) & \text{Weight } (a_i) \\
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
\text{Astoria} & 432 & 787 \\
\text{Jackson Heights} & 627 & 347 \\
\text{Flushing} & 629 & 274 \\
\text{Sunnyside} & 292 & 642 \\
\text{Ditmars} & 978 & 130 \\
\end{array}
\]

Total development capacity (from capacity.csv):

\[
C = 586
\]

##### Mathematical Model

**Decision variables:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} a_i x_i \leq 586
\]

**Where:**

- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $b_i$ = Value for area $i$ (see table above)
- $a_i$ = Weight for area $i$ (see table above)
- $C = 586$ (overall development capacity)

**Complete Formulation:**

\[
\begin{align*}
\max\quad & 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} \\
& + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} \\
& + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \\
\text{s.t.}\quad & 954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} \\
& + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} \\
& + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]