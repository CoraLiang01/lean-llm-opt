Let $I$ be the set of all TABLET products listed below, indexed by their "Product Name". For each $i \in I$, let:

- $r_i$ = Revenue for product $i$
- $d_i$ = Demand for product $i$
- $s_i$ = Initial Inventory for product $i$
- $x_i$ = Number of units of product $i$ to fulfill (decision variable)

The model is:

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Subject to:**

For each $i \in I$:
\[
0 \leq x_i \leq \min\{d_i,\, s_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

**Data:**

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

For each $i$, $x_i$ is an integer such that $0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\}$.

**Decision variables:**
\[
x_i = \text{number of units of TABLET product } i \text{ to fulfill}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]