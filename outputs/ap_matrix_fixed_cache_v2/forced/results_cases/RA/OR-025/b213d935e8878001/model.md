Let $I$ be the set of all products classified as ‘TABLET’, indexed by their "Product Name" as given in the data. For each $i \in I$, let:

- $r_i$ = Revenue for product $i$
- $d_i$ = Demand for product $i$
- $s_i$ = Initial Inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

The model is:

Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to, for each $i \in I$:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \\
x_i \in \mathbb{Z}_{\geq 0}
$$

Where the data is:

| Product Name         | Revenue   | Demand | Initial Inventory |
|----------------------|-----------|--------|------------------|
| TABLET_10084.74      | 10084.74  | 2      | 10               |
| TABLET_12211.86      | 12211.86  | 43     | 300              |
| TABLET_14669.5       | 14669.5   | 6      | 30               |
| TABLET_14745.76      | 14745.76  | 6      | 30               |
| TABLET_14754.24      | 14754.24  | 20     | 100              |
| TABLET_16448.3       | 16448.3   | 22     | 110              |
| TABLET_16448.31      | 16448.31  | 3      | 20               |
| TABLET_20143.22      | 20143.22  | 16     | 80               |
| TABLET_2042.38       | 2042.38   | 2      | 10               |
| TABLET_24915.25      | 24915.25  | 2      | 10               |
| TABLET_24915.26      | 24915.26  | 32     | 160              |
| TABLET_26448.3       | 26448.3   | 14     | 70               |
| TABLET_27042.38      | 27042.38  | 2      | 10               |
| TABLET_30000.0       | 30000.0   | 2      | 10               |
| TABLET_33397.46      | 33397.46  | 6      | 30               |
| TABLET_33398.3       | 33398.3   | 2      | 10               |
| TABLET_48567.8       | 48567.8   | 6      | 30               |
| TABLET_48644.07      | 48644.07  | 2      | 10               |
| TABLET_50262.72      | 50262.72  | 6      | 30               |
| TABLET_53736.44      | 53736.44  | 6      | 30               |
| TABLET_6957.62       | 6957.62   | 6      | 40               |
| TABLET_6957.63       | 6957.63   | 12     | 80               |
| TABLET_7550.84       | 7550.84   | 60     | 300              |
| TABLET_7550.85       | 7550.85   | 8      | 40               |
| TABLET_9584.74       | 9584.74   | 8      | 40               |
| TABLET_9661.02       | 9661.02   | 38     | 190              |
| TABLET_9669.5        | 9669.5    | 4      | 20               |

So, for each $i$ (each row above):

$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad x_i \in \mathbb{Z}_{\geq 0}
$$

Objective:
$$
\max \left(
10084.74\, x_{\text{TABLET\_10084.74}} +
12211.86\, x_{\text{TABLET\_12211.86}} +
14669.5\, x_{\text{TABLET\_14669.5}} +
14745.76\, x_{\text{TABLET\_14745.76}} +
14754.24\, x_{\text{TABLET\_14754.24}} +
16448.3\, x_{\text{TABLET\_16448.3}} +
16448.31\, x_{\text{TABLET\_16448.31}} +
20143.22\, x_{\text{TABLET\_20143.22}} +
2042.38\, x_{\text{TABLET\_2042.38}} +
24915.25\, x_{\text{TABLET\_24915.25}} +
24915.26\, x_{\text{TABLET\_24915.26}} +
26448.3\, x_{\text{TABLET\_26448.3}} +
27042.38\, x_{\text{TABLET\_27042.38}} +
30000.0\, x_{\text{TABLET\_30000.0}} +
33397.46\, x_{\text{TABLET\_33397.46}} +
33398.3\, x_{\text{TABLET\_33398.3}} +
48567.8\, x_{\text{TABLET\_48567.8}} +
48644.07\, x_{\text{TABLET\_48644.07}} +
50262.72\, x_{\text{TABLET\_50262.72}} +
53736.44\, x_{\text{TABLET\_53736.44}} +
6957.62\, x_{\text{TABLET\_6957.62}} +
6957.63\, x_{\text{TABLET\_6957.63}} +
7550.84\, x_{\text{TABLET\_7550.84}} +
7550.85\, x_{\text{TABLET\_7550.85}} +
9584.74\, x_{\text{TABLET\_9584.74}} +
9661.02\, x_{\text{TABLET\_9661.02}} +
9669.5\, x_{\text{TABLET\_9669.5}}
\right)
$$

Subject to, for each product $i$:

$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad x_i \in \mathbb{Z}_{\geq 0}
$$

where $x_i$ is the number of units of product $i$ fulfilled.