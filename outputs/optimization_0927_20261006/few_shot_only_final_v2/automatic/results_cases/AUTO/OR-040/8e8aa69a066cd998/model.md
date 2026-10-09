Let $x_i$ be the integer number of development units in area $i$ per day, for each area $i$ listed below.

**Objective:**
\[
\max \sum_{i} v_i x_i
\]
where $v_i$ is the "Value" for area $i$.

**Subject to:**
\[
\sum_{i} w_i x_i \leq 4466
\]
where $w_i$ is the "Weight" for area $i$.

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Parameters and Indices (from products.csv, in source order):**

| $i$ | ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|---------------|
| 1   | Queens              | 443           | 104           |
| 2   | Brooklyn            | 522           | 368           |
| 3   | Manhattan           | 300           | 483           |
| 4   | Bronx               | 767           | 165           |
| 5   | Staten Island       | 300           | 105           |
| 6   | Harlem              | 309           | 123           |
| 7   | Upper East Side     | 598           | 131           |
| 8   | Lower Manhattan     | 460           | 341           |
| 9   | Midtown             | 318           | 258           |
| 10  | Long Island City    | 126           | 469           |
| 11  | Williamsburg        | 593           | 387           |
| 12  | Bushwick            | 871           | 425           |
| 13  | Flatbush            | 858           | 482           |
| 14  | Greenpoint          | 321           | 495           |
| 15  | Park Slope          | 275           | 305           |
| 16  | Astoria             | 700           | 377           |
| 17  | Jackson Heights     | 685           | 318           |
| 18  | Flushing            | 940           | 56            |
| 19  | Sunnyside           | 522           | 213           |
| 20  | Ditmars             | 763           | 472           |

**Complete Model:**

\[
\max \Big(
443\,x_1 + 522\,x_2 + 300\,x_3 + 767\,x_4 + 300\,x_5 + 309\,x_6 + 598\,x_7 + 460\,x_8 + 318\,x_9 + 126\,x_{10} + 593\,x_{11} + 871\,x_{12} + 858\,x_{13} + 321\,x_{14} + 275\,x_{15} + 700\,x_{16} + 685\,x_{17} + 940\,x_{18} + 522\,x_{19} + 763\,x_{20}
\Big)
\]

subject to

\[
104\,x_1 + 368\,x_2 + 483\,x_3 + 165\,x_4 + 105\,x_5 + 123\,x_6 + 131\,x_7 + 341\,x_8 + 258\,x_9 + 469\,x_{10} + 387\,x_{11} + 425\,x_{12} + 482\,x_{13} + 495\,x_{14} + 305\,x_{15} + 377\,x_{16} + 318\,x_{17} + 56\,x_{18} + 213\,x_{19} + 472\,x_{20} \leq 4466
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 20
\]