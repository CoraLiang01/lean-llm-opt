Let $x_i$ be the integer number of development units per day in area $i$, where $i$ indexes the following areas:

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

Parameters:

- Benefit coefficients $b_i$ (from Value column):

\[
\begin{array}{ll}
b_1 = 443 & (\text{Queens}) \\
b_2 = 522 & (\text{Brooklyn}) \\
b_3 = 300 & (\text{Manhattan}) \\
b_4 = 767 & (\text{Bronx}) \\
b_5 = 300 & (\text{Staten Island}) \\
b_6 = 309 & (\text{Harlem}) \\
b_7 = 598 & (\text{Upper East Side}) \\
b_8 = 460 & (\text{Lower Manhattan}) \\
b_9 = 318 & (\text{Midtown}) \\
b_{10} = 126 & (\text{Long Island City}) \\
b_{11} = 593 & (\text{Williamsburg}) \\
b_{12} = 871 & (\text{Bushwick}) \\
b_{13} = 858 & (\text{Flatbush}) \\
b_{14} = 321 & (\text{Greenpoint}) \\
b_{15} = 275 & (\text{Park Slope}) \\
b_{16} = 700 & (\text{Astoria}) \\
b_{17} = 685 & (\text{Jackson Heights}) \\
b_{18} = 940 & (\text{Flushing}) \\
b_{19} = 522 & (\text{Sunnyside}) \\
b_{20} = 763 & (\text{Ditmars}) \\
\end{array}
\]

- Development unit weights $w_i$ (from Weight column):

\[
\begin{array}{ll}
w_1 = 104 & (\text{Queens}) \\
w_2 = 368 & (\text{Brooklyn}) \\
w_3 = 483 & (\text{Manhattan}) \\
w_4 = 165 & (\text{Bronx}) \\
w_5 = 105 & (\text{Staten Island}) \\
w_6 = 123 & (\text{Harlem}) \\
w_7 = 131 & (\text{Upper East Side}) \\
w_8 = 341 & (\text{Lower Manhattan}) \\
w_9 = 258 & (\text{Midtown}) \\
w_{10} = 469 & (\text{Long Island City}) \\
w_{11} = 387 & (\text{Williamsburg}) \\
w_{12} = 425 & (\text{Bushwick}) \\
w_{13} = 482 & (\text{Flatbush}) \\
w_{14} = 495 & (\text{Greenpoint}) \\
w_{15} = 305 & (\text{Park Slope}) \\
w_{16} = 377 & (\text{Astoria}) \\
w_{17} = 318 & (\text{Jackson Heights}) \\
w_{18} = 56 & (\text{Flushing}) \\
w_{19} = 213 & (\text{Sunnyside}) \\
w_{20} = 472 & (\text{Ditmars}) \\
\end{array}
\]

- Total development capacity: $C = 4466$

Model:

Maximize total benefit:
\[
\max \sum_{i=1}^{20} b_i x_i
\]

Subject to the overall capacity constraint:
\[
\sum_{i=1}^{20} w_i x_i \leq 4466
\]

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 20
\]

Where $x_i$ is the integer number of development units per day in area $i$ as listed above.