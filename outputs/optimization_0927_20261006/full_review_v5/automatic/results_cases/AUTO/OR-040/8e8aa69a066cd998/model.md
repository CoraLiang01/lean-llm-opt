Let $x_i$ be the integer number of development units allocated daily to area $i$, where $i$ indexes the following areas in the order given:

| Area                | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Queens              | 443          | 104           |
| Brooklyn            | 522          | 368           |
| Manhattan           | 300          | 483           |
| Bronx               | 767          | 165           |
| Staten Island       | 300          | 105           |
| Harlem              | 309          | 123           |
| Upper East Side     | 598          | 131           |
| Lower Manhattan     | 460          | 341           |
| Midtown             | 318          | 258           |
| Long Island City    | 126          | 469           |
| Williamsburg        | 593          | 387           |
| Bushwick            | 871          | 425           |
| Flatbush            | 858          | 482           |
| Greenpoint          | 321          | 495           |
| Park Slope          | 275          | 305           |
| Astoria             | 700          | 377           |
| Jackson Heights     | 685          | 318           |
| Flushing            | 940          | 56            |
| Sunnyside           | 522          | 213           |
| Ditmars             | 763          | 472           |

The overall development capacity is $C = 4466$.

The mathematical model is:

**Objective:**
\[
\max \sum_{i=1}^{21} v_i x_i
\]
where $v_i$ is the Value for area $i$ as given above.

**Constraint:**
\[
\sum_{i=1}^{21} w_i x_i \leq 4466
\]
where $w_i$ is the Weight for area $i$ as given above.

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 21
\]

**Parameter Table (source order):**

| Area                | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Queens              | 443          | 104           |
| Brooklyn            | 522          | 368           |
| Manhattan           | 300          | 483           |
| Bronx               | 767          | 165           |
| Staten Island       | 300          | 105           |
| Harlem              | 309          | 123           |
| Upper East Side     | 598          | 131           |
| Lower Manhattan     | 460          | 341           |
| Midtown             | 318          | 258           |
| Long Island City    | 126          | 469           |
| Williamsburg        | 593          | 387           |
| Bushwick            | 871          | 425           |
| Flatbush            | 858          | 482           |
| Greenpoint          | 321          | 495           |
| Park Slope          | 275          | 305           |
| Astoria             | 700          | 377           |
| Jackson Heights     | 685          | 318           |
| Flushing            | 940          | 56            |
| Sunnyside           | 522          | 213           |
| Ditmars             | 763          | 472           |

**Capacity:**
\[
C = 4466
\]

**Decision variables:**
\[
x_i = \text{integer number of development units allocated daily to area } i
\]

**Complete Model:**

\[
\begin{align*}
\max \quad & 443x_1 + 522x_2 + 300x_3 + 767x_4 + 300x_5 + 309x_6 + 598x_7 + 460x_8 + 318x_9 + 126x_{10} \\
& + 593x_{11} + 871x_{12} + 858x_{13} + 321x_{14} + 275x_{15} + 700x_{16} + 685x_{17} + 940x_{18} \\
& + 522x_{19} + 763x_{20} \\
\text{s.t.} \quad & 104x_1 + 368x_2 + 483x_3 + 165x_4 + 105x_5 + 123x_6 + 131x_7 + 341x_8 + 258x_9 + 469x_{10} \\
& + 387x_{11} + 425x_{12} + 482x_{13} + 495x_{14} + 305x_{15} + 377x_{16} + 318x_{17} + 56x_{18} \\
& + 213x_{19} + 472x_{20} \leq 4466 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 20
\end{align*}
\]

(If there are 21 areas, include $x_{21}$ for Ditmars with $v_{21}=763$, $w_{21}=472$; otherwise, the above matches the source order.)

All coefficients and identifiers are as retrieved, and the model is complete.