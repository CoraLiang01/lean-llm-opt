Let $I$ be the set of TABLET models:

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

Let $x_i$ be the number of units of TABLET model $i$ to fulfill (decision variable, integer, $x_i \geq 0$).

Parameters for each $i \in I$ (from source order):

| Product Name         | Revenue   | Demand | Initial Inventory |
|---------------------|-----------|--------|------------------|
| TABLET_10084.74     | 10084.74  | 2      | 10               |
| TABLET_12211.86     | 12211.86  | 43     | 300              |
| TABLET_14669.5      | 14669.5   | 6      | 30               |
| TABLET_14745.76     | 14745.76  | 6      | 30               |
| TABLET_14754.24     | 14754.24  | 20     | 100              |
| TABLET_16448.3      | 16448.3   | 22     | 110              |
| TABLET_16448.31     | 16448.31  | 3      | 20               |
| TABLET_20143.22     | 20143.22  | 16     | 80               |
| TABLET_2042.38      | 2042.38   | 2      | 10               |
| TABLET_24915.25     | 24915.25  | 2      | 10               |
| TABLET_24915.26     | 24915.26  | 32     | 160              |
| TABLET_26448.3      | 26448.3   | 14     | 70               |
| TABLET_27042.38     | 27042.38  | 2      | 10               |
| TABLET_30000.0      | 30000.0   | 2      | 10               |
| TABLET_33397.46     | 33397.46  | 6      | 30               |
| TABLET_33398.3      | 33398.3   | 2      | 10               |
| TABLET_48567.8      | 48567.8   | 6      | 30               |
| TABLET_48644.07     | 48644.07  | 2      | 10               |
| TABLET_50262.72     | 50262.72  | 6      | 30               |
| TABLET_53736.44     | 53736.44  | 6      | 30               |
| TABLET_6957.62      | 6957.62   | 6      | 40               |
| TABLET_6957.63      | 6957.63   | 12     | 80               |
| TABLET_7550.84      | 7550.84   | 60     | 300              |
| TABLET_7550.85      | 7550.85   | 8      | 40               |
| TABLET_9584.74      | 9584.74   | 8      | 40               |
| TABLET_9661.02      | 9661.02   | 38     | 190              |
| TABLET_9669.5       | 9669.5    | 4      | 20               |

Model:

Maximize total revenue:
\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to, for each $i \in I$:
\[
0 \leq x_i \leq \min(\text{Demand}_i, \text{Initial Inventory}_i)
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Where all coefficients and identifiers are as listed above, in source order.