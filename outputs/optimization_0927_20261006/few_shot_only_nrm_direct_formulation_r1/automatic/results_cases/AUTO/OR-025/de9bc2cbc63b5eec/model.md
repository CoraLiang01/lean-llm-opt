**Index Set:**  
Let $I$ be the set of all 'TABLET' products (in source order):

\[
I = \left\{
\begin{array}{ll}
\text{TABLET\_10084.74} \\
\text{TABLET\_12211.86} \\
\text{TABLET\_14669.5} \\
\text{TABLET\_14745.76} \\
\text{TABLET\_14754.24} \\
\text{TABLET\_16448.3} \\
\text{TABLET\_16448.31} \\
\text{TABLET\_20143.22} \\
\text{TABLET\_2042.38} \\
\text{TABLET\_24915.25} \\
\text{TABLET\_24915.26} \\
\text{TABLET\_26448.3} \\
\text{TABLET\_27042.38} \\
\text{TABLET\_30000.0} \\
\text{TABLET\_33397.46} \\
\text{TABLET\_33398.3} \\
\text{TABLET\_48567.8} \\
\text{TABLET\_48644.07} \\
\text{TABLET\_50262.72} \\
\text{TABLET\_53736.44} \\
\text{TABLET\_6957.62} \\
\text{TABLET\_6957.63} \\
\text{TABLET\_7550.84} \\
\text{TABLET\_7550.85} \\
\text{TABLET\_9584.74} \\
\text{TABLET\_9661.02} \\
\text{TABLET\_9669.5}
\end{array}
\right\}
\]

**Parameters:**  
For each $i \in I$ (in source order):

| $i$                        | Revenue $A_i$ | Demand $d_i$ | Initial Inventory $I_i$ |
|----------------------------|--------------:|-------------:|-----------------------:|
| TABLET_10084.74            | 10084.74      | 2            | 10                    |
| TABLET_12211.86            | 12211.86      | 43           | 300                   |
| TABLET_14669.5             | 14669.5       | 6            | 30                    |
| TABLET_14745.76            | 14745.76      | 6            | 30                    |
| TABLET_14754.24            | 14754.24      | 20           | 100                   |
| TABLET_16448.3             | 16448.3       | 22           | 110                   |
| TABLET_16448.31            | 16448.31      | 3            | 20                    |
| TABLET_20143.22            | 20143.22      | 16           | 80                    |
| TABLET_2042.38             | 2042.38       | 2            | 10                    |
| TABLET_24915.25            | 24915.25      | 2            | 10                    |
| TABLET_24915.26            | 24915.26      | 32           | 160                   |
| TABLET_26448.3             | 26448.3       | 14           | 70                    |
| TABLET_27042.38            | 27042.38      | 2            | 10                    |
| TABLET_30000.0             | 30000.0       | 2            | 10                    |
| TABLET_33397.46            | 33397.46      | 6            | 30                    |
| TABLET_33398.3             | 33398.3       | 2            | 10                    |
| TABLET_48567.8             | 48567.8       | 6            | 30                    |
| TABLET_48644.07            | 48644.07      | 2            | 10                    |
| TABLET_50262.72            | 50262.72      | 6            | 30                    |
| TABLET_53736.44            | 53736.44      | 6            | 30                    |
| TABLET_6957.62             | 6957.62       | 6            | 40                    |
| TABLET_6957.63             | 6957.63       | 12           | 80                    |
| TABLET_7550.84             | 7550.84       | 60           | 300                   |
| TABLET_7550.85             | 7550.85       | 8            | 40                    |
| TABLET_9584.74             | 9584.74       | 8            | 40                    |
| TABLET_9661.02             | 9661.02       | 38           | 190                   |
| TABLET_9669.5              | 9669.5        | 4            | 20                    |

**Decision Variables:**  
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
For each $i \in I$:
\[
\begin{align*}
x_i &\leq d_i \\
x_i &\leq I_i \\
x_i &\geq 0 \\
x_i &\in \mathbb{Z}
\end{align*}
\]

**Full Model (Numerical):**

\[
\begin{align*}
\max \quad & 10084.74\, x_{\text{TABLET\_10084.74}}
+ 12211.86\, x_{\text{TABLET\_12211.86}}
+ 14669.5\, x_{\text{TABLET\_14669.5}}
+ 14745.76\, x_{\text{TABLET\_14745.76}} \\
& + 14754.24\, x_{\text{TABLET\_14754.24}}
+ 16448.3\, x_{\text{TABLET\_16448.3}}
+ 16448.31\, x_{\text{TABLET\_16448.31}}
+ 20143.22\, x_{\text{TABLET\_20143.22}} \\
& + 2042.38\, x_{\text{TABLET\_2042.38}}
+ 24915.25\, x_{\text{TABLET\_24915.25}}
+ 24915.26\, x_{\text{TABLET\_24915.26}}
+ 26448.3\, x_{\text{TABLET\_26448.3}} \\
& + 27042.38\, x_{\text{TABLET\_27042.38}}
+ 30000.0\, x_{\text{TABLET\_30000.0}}
+ 33397.46\, x_{\text{TABLET\_33397.46}}
+ 33398.3\, x_{\text{TABLET\_33398.3}} \\
& + 48567.8\, x_{\text{TABLET\_48567.8}}
+ 48644.07\, x_{\text{TABLET\_48644.07}}
+ 50262.72\, x_{\text{TABLET\_50262.72}}
+ 53736.44\, x_{\text{TABLET\_53736.44}} \\
& + 6957.62\, x_{\text{TABLET\_6957.62}}
+ 6957.63\, x_{\text{TABLET\_6957.63}}
+ 7550.84\, x_{\text{TABLET\_7550.84}}
+ 7550.85\, x_{\text{TABLET\_7550.85}} \\
& + 9584.74\, x_{\text{TABLET\_9584.74}}
+ 9661.02\, x_{\text{TABLET\_9661.02}}
+ 9669.5\, x_{\text{TABLET\_9669.5}}
\end{align*}
\]

Subject to, for each $i \in I$ (in source order):

\[
\begin{align*}
0 \leq x_{\text{TABLET\_10084.74}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_12211.86}} &\leq \min\{43, 300\} = 43 \\
0 \leq x_{\text{TABLET\_14669.5}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_14745.76}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_14754.24}} &\leq \min\{20, 100\} = 20 \\
0 \leq x_{\text{TABLET\_16448.3}} &\leq \min\{22, 110\} = 22 \\
0 \leq x_{\text{TABLET\_16448.31}} &\leq \min\{3, 20\} = 3 \\
0 \leq x_{\text{TABLET\_20143.22}} &\leq \min\{16, 80\} = 16 \\
0 \leq x_{\text{TABLET\_2042.38}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_24915.25}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_24915.26}} &\leq \min\{32, 160\} = 32 \\
0 \leq x_{\text{TABLET\_26448.3}} &\leq \min\{14, 70\} = 14 \\
0 \leq x_{\text{TABLET\_27042.38}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_30000.0}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_33397.46}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_33398.3}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_48567.8}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_48644.07}} &\leq \min\{2, 10\} = 2 \\
0 \leq x_{\text{TABLET\_50262.72}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_53736.44}} &\leq \min\{6, 30\} = 6 \\
0 \leq x_{\text{TABLET\_6957.62}} &\leq \min\{6, 40\} = 6 \\
0 \leq x_{\text{TABLET\_6957.63}} &\leq \min\{12, 80\} = 12 \\
0 \leq x_{\text{TABLET\_7550.84}} &\leq \min\{60, 300\} = 60 \\
0 \leq x_{\text{TABLET\_7550.85}} &\leq \min\{8, 40\} = 8 \\
0 \leq x_{\text{TABLET\_9584.74}} &\leq \min\{8, 40\} = 8 \\
0 \leq x_{\text{TABLET\_9661.02}} &\leq \min\{38, 190\} = 38 \\
0 \leq x_{\text{TABLET\_9669.5}} &\leq \min\{4, 20\} = 4 \\
\end{align*}
\]

and

\[
x_i \in \mathbb{Z}, \quad \forall i \in I
\]

**Retrieved Information (for code generation):**

- Product Names (in source order):  
  TABLET_10084.74, TABLET_12211.86, TABLET_14669.5, TABLET_14745.76, TABLET_14754.24, TABLET_16448.3, TABLET_16448.31, TABLET_20143.22, TABLET_2042.38, TABLET_24915.25, TABLET_24915.26, TABLET_26448.3, TABLET_27042.38, TABLET_30000.0, TABLET_33397.46, TABLET_33398.3, TABLET_48567.8, TABLET_48644.07, TABLET_50262.72, TABLET_53736.44, TABLET_6957.62, TABLET_6957.63, TABLET_7550.84, TABLET_7550.85, TABLET_9584.74, TABLET_9661.02, TABLET_9669.5

- Revenue vector $A_i$ (in source order):  
  [10084.74, 12211.86, 14669.5, 14745.76, 14754.24, 16448.3, 16448.31, 20143.22, 2042.38, 24915.25, 24915.26, 26448.3, 27042.38, 30000.0, 33397.46, 33398.3, 48567.8, 48644.07, 50262.72, 53736.44, 6957.62, 6957.63, 7550.84, 7550.85, 9584.74, 9661.02, 9669.5]

- Demand vector $d_i$ (in source order):  
  [2, 43, 6, 6, 20, 22, 3, 16, 2, 2, 32, 14, 2, 2, 6, 2, 6, 2, 6, 6, 6, 12, 60, 8, 8, 38, 4]

- Initial Inventory vector $I_i$ (in source order):  
  [10, 300, 30, 30, 100, 110, 20, 80, 10, 10, 160, 70, 10, 10, 30, 10, 30, 10, 30, 30, 40, 80, 300, 40, 40, 190, 20]

**Variable domains:**  
For each $i \in I$: $x_i \in \mathbb{Z}_+, \ 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue from fulfilled units of all TABLET products.