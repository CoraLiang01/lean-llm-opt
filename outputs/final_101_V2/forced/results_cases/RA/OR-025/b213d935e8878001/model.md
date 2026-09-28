Let $I$ be the set of all TABLET products:

\[
I = \{
\text{TABLET\_10084.74},
\text{TABLET\_12211.86},
\text{TABLET\_14669.5},
\text{TABLET\_14745.76},
\text{TABLET\_14754.24},
\text{TABLET\_16448.3},
\text{TABLET\_16448.31},
\text{TABLET\_20143.22},
\text{TABLET\_2042.38},
\text{TABLET\_24915.25},
\text{TABLET\_24915.26},
\text{TABLET\_26448.3},
\text{TABLET\_27042.38},
\text{TABLET\_30000.0},
\text{TABLET\_33397.46},
\text{TABLET\_33398.3},
\text{TABLET\_48567.8},
\text{TABLET\_48644.07},
\text{TABLET\_50262.72},
\text{TABLET\_53736.44},
\text{TABLET\_6957.62},
\text{TABLET\_6957.63},
\text{TABLET\_7550.84},
\text{TABLET\_7550.85},
\text{TABLET\_9584.74},
\text{TABLET\_9661.02},
\text{TABLET\_9669.5}
\}
\]

Let $x_i$ be the number of units of product $i$ to fulfill, for each $i \in I$.

Parameters for each $i \in I$:

- $r_i$: Revenue per unit
- $s_i$: Initial Inventory
- $d_i$: Demand

Given data:

| $i$                   | $r_i$    | $s_i$ | $d_i$ |
|-----------------------|----------|-------|-------|
| TABLET_10084.74       | 10084.74 | 10    | 2     |
| TABLET_12211.86       | 12211.86 | 300   | 43    |
| TABLET_14669.5        | 14669.5  | 30    | 6     |
| TABLET_14745.76       | 14745.76 | 30    | 6     |
| TABLET_14754.24       | 14754.24 | 100   | 20    |
| TABLET_16448.3        | 16448.3  | 110   | 22    |
| TABLET_16448.31       | 16448.31 | 20    | 3     |
| TABLET_20143.22       | 20143.22 | 80    | 16    |
| TABLET_2042.38        | 2042.38  | 10    | 2     |
| TABLET_24915.25       | 24915.25 | 10    | 2     |
| TABLET_24915.26       | 24915.26 | 160   | 32    |
| TABLET_26448.3        | 26448.3  | 70    | 14    |
| TABLET_27042.38       | 27042.38 | 10    | 2     |
| TABLET_30000.0        | 30000.0  | 10    | 2     |
| TABLET_33397.46       | 33397.46 | 30    | 6     |
| TABLET_33398.3        | 33398.3  | 10    | 2     |
| TABLET_48567.8        | 48567.8  | 30    | 6     |
| TABLET_48644.07       | 48644.07 | 10    | 2     |
| TABLET_50262.72       | 50262.72 | 30    | 6     |
| TABLET_53736.44       | 53736.44 | 30    | 6     |
| TABLET_6957.62        | 6957.62  | 40    | 6     |
| TABLET_6957.63        | 6957.63  | 80    | 12    |
| TABLET_7550.84        | 7550.84  | 300   | 60    |
| TABLET_7550.85        | 7550.85  | 40    | 8     |
| TABLET_9584.74        | 9584.74  | 40    | 8     |
| TABLET_9661.02        | 9661.02  | 190   | 38    |
| TABLET_9669.5         | 9669.5   | 20    | 4     |

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to, for all $i \in I$:
\[
0 \leq x_i \leq \min\{s_i, d_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Where:
- $x_i$ = number of units of product $i$ fulfilled (decision variable, nonnegative integer)
- $r_i$ = revenue per unit of product $i$
- $s_i$ = initial inventory of product $i$
- $d_i$ = demand for product $i$

All coefficients and identifiers are as retrieved above.