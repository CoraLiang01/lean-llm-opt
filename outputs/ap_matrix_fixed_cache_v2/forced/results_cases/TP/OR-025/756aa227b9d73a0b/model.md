Let $I$ be the set of all ‘TABLET’ smartphone models, indexed in source order as follows:

\[
\begin{array}{llll}
\text{Index} & \text{Product Name} & \text{Revenue}_i & \text{Initial Inventory}_i & \text{Demand}_i \\
1 & \text{TABLET\_10084.74} & 10084.74 & 10 & 2 \\
2 & \text{TABLET\_12211.86} & 12211.86 & 300 & 43 \\
3 & \text{TABLET\_14669.5} & 14669.5 & 30 & 6 \\
4 & \text{TABLET\_14745.76} & 14745.76 & 30 & 6 \\
5 & \text{TABLET\_14754.24} & 14754.24 & 100 & 20 \\
6 & \text{TABLET\_16448.3} & 16448.3 & 110 & 22 \\
7 & \text{TABLET\_16448.31} & 16448.31 & 20 & 3 \\
8 & \text{TABLET\_20143.22} & 20143.22 & 80 & 16 \\
9 & \text{TABLET\_2042.38} & 2042.38 & 10 & 2 \\
10 & \text{TABLET\_24915.25} & 24915.25 & 10 & 2 \\
11 & \text{TABLET\_24915.26} & 24915.26 & 160 & 32 \\
12 & \text{TABLET\_26448.3} & 26448.3 & 70 & 14 \\
13 & \text{TABLET\_27042.38} & 27042.38 & 10 & 2 \\
14 & \text{TABLET\_30000.0} & 30000.0 & 10 & 2 \\
15 & \text{TABLET\_33397.46} & 33397.46 & 30 & 6 \\
16 & \text{TABLET\_33398.3} & 33398.3 & 10 & 2 \\
17 & \text{TABLET\_48567.8} & 48567.8 & 30 & 6 \\
18 & \text{TABLET\_48644.07} & 48644.07 & 10 & 2 \\
19 & \text{TABLET\_50262.72} & 50262.72 & 30 & 6 \\
20 & \text{TABLET\_53736.44} & 53736.44 & 30 & 6 \\
21 & \text{TABLET\_6957.62} & 6957.62 & 40 & 6 \\
22 & \text{TABLET\_6957.63} & 6957.63 & 80 & 12 \\
23 & \text{TABLET\_7550.84} & 7550.84 & 300 & 60 \\
24 & \text{TABLET\_7550.85} & 7550.85 & 40 & 8 \\
25 & \text{TABLET\_9584.74} & 9584.74 & 40 & 8 \\
26 & \text{TABLET\_9661.02} & 9661.02 & 190 & 38 \\
27 & \text{TABLET\_9669.5} & 9669.5 & 20 & 4 \\
\end{array}
\]

Decision variables:
\[
x_i = \text{number of units of model } i \text{ fulfilled},\quad x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

Objective:
\[
\max \sum_{i\in I} \text{Revenue}_i \cdot x_i
\]

Subject to:
\[
0 \leq x_i \leq \min(\text{Initial Inventory}_i, \text{Demand}_i),\quad \forall i \in I
\]

Explicitly, for each $i$ (in source order):

\[
\begin{align*}
0 \leq x_1 &\leq 2 \\
0 \leq x_2 &\leq 43 \\
0 \leq x_3 &\leq 6 \\
0 \leq x_4 &\leq 6 \\
0 \leq x_5 &\leq 20 \\
0 \leq x_6 &\leq 22 \\
0 \leq x_7 &\leq 3 \\
0 \leq x_8 &\leq 16 \\
0 \leq x_9 &\leq 2 \\
0 \leq x_{10} &\leq 2 \\
0 \leq x_{11} &\leq 32 \\
0 \leq x_{12} &\leq 14 \\
0 \leq x_{13} &\leq 2 \\
0 \leq x_{14} &\leq 2 \\
0 \leq x_{15} &\leq 6 \\
0 \leq x_{16} &\leq 2 \\
0 \leq x_{17} &\leq 6 \\
0 \leq x_{18} &\leq 2 \\
0 \leq x_{19} &\leq 6 \\
0 \leq x_{20} &\leq 6 \\
0 \leq x_{21} &\leq 6 \\
0 \leq x_{22} &\leq 12 \\
0 \leq x_{23} &\leq 60 \\
0 \leq x_{24} &\leq 8 \\
0 \leq x_{25} &\leq 8 \\
0 \leq x_{26} &\leq 38 \\
0 \leq x_{27} &\leq 4 \\
\end{align*}
\]

All $x_i$ are integer.

Summary of parameters (in source order):

| $i$ | Product Name         | Revenue$_i$ | Initial Inventory$_i$ | Demand$_i$ | $\min(\text{Initial Inventory}_i, \text{Demand}_i)$ |
|-----|---------------------|-------------|----------------------|------------|-----------------------------------------------------|
| 1   | TABLET_10084.74     | 10084.74    | 10                   | 2          | 2                                                   |
| 2   | TABLET_12211.86     | 12211.86    | 300                  | 43         | 43                                                  |
| 3   | TABLET_14669.5      | 14669.5     | 30                   | 6          | 6                                                   |
| 4   | TABLET_14745.76     | 14745.76    | 30                   | 6          | 6                                                   |
| 5   | TABLET_14754.24     | 14754.24    | 100                  | 20         | 20                                                  |
| 6   | TABLET_16448.3      | 16448.3     | 110                  | 22         | 22                                                  |
| 7   | TABLET_16448.31     | 16448.31    | 20                   | 3          | 3                                                   |
| 8   | TABLET_20143.22     | 20143.22    | 80                   | 16         | 16                                                  |
| 9   | TABLET_2042.38      | 2042.38     | 10                   | 2          | 2                                                   |
| 10  | TABLET_24915.25     | 24915.25    | 10                   | 2          | 2                                                   |
| 11  | TABLET_24915.26     | 24915.26    | 160                  | 32         | 32                                                  |
| 12  | TABLET_26448.3      | 26448.3     | 70                   | 14         | 14                                                  |
| 13  | TABLET_27042.38     | 27042.38    | 10                   | 2          | 2                                                   |
| 14  | TABLET_30000.0      | 30000.0     | 10                   | 2          | 2                                                   |
| 15  | TABLET_33397.46     | 33397.46    | 30                   | 6          | 6                                                   |
| 16  | TABLET_33398.3      | 33398.3     | 10                   | 2          | 2                                                   |
| 17  | TABLET_48567.8      | 48567.8     | 30                   | 6          | 6                                                   |
| 18  | TABLET_48644.07     | 48644.07    | 10                   | 2          | 2                                                   |
| 19  | TABLET_50262.72     | 50262.72    | 30                   | 6          | 6                                                   |
| 20  | TABLET_53736.44     | 53736.44    | 30                   | 6          | 6                                                   |
| 21  | TABLET_6957.62      | 6957.62     | 40                   | 6          | 6                                                   |
| 22  | TABLET_6957.63      | 6957.63     | 80                   | 12         | 12                                                  |
| 23  | TABLET_7550.84      | 7550.84     | 300                  | 60         | 60                                                  |
| 24  | TABLET_7550.85      | 7550.85     | 40                   | 8          | 8                                                   |
| 25  | TABLET_9584.74      | 9584.74     | 40                   | 8          | 8                                                   |
| 26  | TABLET_9661.02      | 9661.02     | 190                  | 38         | 38                                                  |
| 27  | TABLET_9669.5       | 9669.5      | 20                   | 4          | 4                                                   |

Where $x_i$ is integer for all $i$.

Complete model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{27} \text{Revenue}_i \cdot x_i \\
\text{s.t.}\quad & 0 \leq x_i \leq \min(\text{Initial Inventory}_i, \text{Demand}_i),\quad x_i \in \mathbb{Z}_+,\quad \forall i=1,\ldots,27
\end{align*}
\]