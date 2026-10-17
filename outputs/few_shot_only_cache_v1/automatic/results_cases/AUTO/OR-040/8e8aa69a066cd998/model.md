Let $I$ be the set of areas (indexed by the ProductName column in products.csv):

\[
I = \{\text{Queens},\ \text{Brooklyn},\ \text{Manhattan},\ \text{Bronx},\ \text{Staten Island},\ \text{Harlem},\ \text{Upper East Side},\ \text{Lower Manhattan},\ \text{Midtown},\ \text{Long Island City},\ \text{Williamsburg},\ \text{Bushwick},\ \text{Flatbush},\ \text{Greenpoint},\ \text{Park Slope},\ \text{Astoria},\ \text{Jackson Heights},\ \text{Flushing},\ \text{Sunnyside},\ \text{Ditmars}\}
\]

Let $x_i$ be the integer variable representing the daily scale of development in area $i \in I$.

Parameters (from products.csv):

\[
\begin{align*}
&\text{Value}_i: \text{Benefit coefficient for area } i \\
&\text{Weight}_i: \text{Development units consumed per unit in area } i \\
\end{align*}
\]

Total development capacity (from capacity.csv):

\[
C = 4466
\]

The mathematical model is:

\[
\textbf{Maximize} \quad \sum_{i \in I} \text{Value}_i \cdot x_i
\]

\[
\textbf{Subject to:}
\]
\[
\sum_{i \in I} \text{Weight}_i \cdot x_i \leq 4466
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Where:

\[
\begin{array}{lll}
\text{Area} & \text{Value}_i & \text{Weight}_i \\
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

All variables $x_i$ are nonnegative integers.