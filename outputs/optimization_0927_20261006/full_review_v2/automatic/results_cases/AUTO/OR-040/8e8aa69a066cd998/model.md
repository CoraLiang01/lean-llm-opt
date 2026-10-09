Let $x_i$ be the integer number of development units allocated daily to area $i$.

Let the set of areas (in source order) be:
\[
\begin{array}{ll}
1: & \text{Queens} \\
2: & \text{Brooklyn} \\
3: & \text{Manhattan} \\
4: & \text{Bronx} \\
5: & \text{Staten Island} \\
6: & \text{Harlem} \\
7: & \text{Upper East Side} \\
8: & \text{Lower Manhattan} \\
9: & \text{Midtown} \\
10: & \text{Long Island City} \\
11: & \text{Williamsburg} \\
12: & \text{Bushwick} \\
13: & \text{Flatbush} \\
14: & \text{Greenpoint} \\
15: & \text{Park Slope} \\
16: & \text{Astoria} \\
17: & \text{Jackson Heights} \\
18: & \text{Flushing} \\
19: & \text{Sunnyside} \\
20: & \text{Ditmars} \\
\end{array}
\]

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{Area} & \text{Benefit Coefficient } (v_i) & \text{Development Unit Weight } (w_i) \\
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

Overall development capacity (from capacity.csv):

\[
C = 4466
\]

Mathematical Model:

\[
\begin{align*}
\max \quad & 443x_{\text{Queens}} + 522x_{\text{Brooklyn}} + 300x_{\text{Manhattan}} + 767x_{\text{Bronx}} + 300x_{\text{Staten Island}} \\
& + 309x_{\text{Harlem}} + 598x_{\text{Upper East Side}} + 460x_{\text{Lower Manhattan}} + 318x_{\text{Midtown}} + 126x_{\text{Long Island City}} \\
& + 593x_{\text{Williamsburg}} + 871x_{\text{Bushwick}} + 858x_{\text{Flatbush}} + 321x_{\text{Greenpoint}} + 275x_{\text{Park Slope}} \\
& + 700x_{\text{Astoria}} + 685x_{\text{Jackson Heights}} + 940x_{\text{Flushing}} + 522x_{\text{Sunnyside}} + 763x_{\text{Ditmars}} \\
\\
\text{s.t.} \quad & 104x_{\text{Queens}} + 368x_{\text{Brooklyn}} + 483x_{\text{Manhattan}} + 165x_{\text{Bronx}} + 105x_{\text{Staten Island}} \\
& + 123x_{\text{Harlem}} + 131x_{\text{Upper East Side}} + 341x_{\text{Lower Manhattan}} + 258x_{\text{Midtown}} + 469x_{\text{Long Island City}} \\
& + 387x_{\text{Williamsburg}} + 425x_{\text{Bushwick}} + 482x_{\text{Flatbush}} + 495x_{\text{Greenpoint}} + 305x_{\text{Park Slope}} \\
& + 377x_{\text{Astoria}} + 318x_{\text{Jackson Heights}} + 56x_{\text{Flushing}} + 213x_{\text{Sunnyside}} + 472x_{\text{Ditmars}} \leq 4466 \\
\\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{all areas listed above in source order}\}
\end{align*}
\]

Where each $x_i$ is the integer number of development units allocated daily to area $i$. All coefficients and area names are as retrieved and in original order.