Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A[i,j]$ be the binary indicator that option $i$ references asset $j$, $i=1,\dots,120$, $j=1,\dots,6$.

Let $\text{Cost}[i]$, $\Delta[i]$, $\Gamma[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ be the per-option coefficients as given below.

Let $\Delta_{\text{initial}} = 0.25$, $\Gamma_{\text{initial}} = 0.08$, $\text{Vega}_{\text{initial}} = 0.17$.

Let $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$.

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
$$

Subject to, for all $i=1,\dots,120$:
\[
\text{MaxShort}[i] \le x_i \le \text{MaxLong}[i] \qquad x_i \in \mathbb{Z}
\]
\[
y_i \ge x_i,\quad y_i \ge -x_i,\quad y_i \ge 0
\]

Greek risk constraints:
\[
\left| \Delta_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \right| \leq 0.06
\]
\[
\left| \Gamma_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \right| \leq 0.05
\]
\[
\left| \text{Vega}_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.07
\]

Or, equivalently, for each Greek $G$:
\[
- \text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_G
\]

All coefficients and bounds are as follows (source order):

#### OptionCharacteristics.csv

| $i$ | Option   | Cost | Delta  | Gamma  | Vega  | MaxLong | MaxShort |
|-----|----------|------|--------|--------|-------|---------|----------|
| 1   | Opt_1    | 9    | -0.54  | 0.12   | 0.10  | 9       | -14      |
| 2   | Opt_2    | 6    | 0.51   | 0.10   | 0.16  | 9       | -14      |
| 3   | Opt_3    | 13   | 0.17   | 0.02   | 0.19  | 10      | -7       |
| 4   | Opt_4    | 10   | -0.24  | 0.03   | 0.18  | 7       | -5       |
| 5   | Opt_5    | 7    | -0.61  | 0.14   | 0.11  | 12      | -9       |
| 6   | Opt_6    | 9    | -0.26  | 0.09   | 0.24  | 5       | -13      |
| 7   | Opt_7    | 12   | -0.24  | 0.01   | 0.20  | 10      | -5       |
| 8   | Opt_8    | 5    | 0.32   | 0.02   | 0.16  | 8       | -7       |
| 9   | Opt_9    | 9    | 0.19   | 0.10   | 0.17  | 5       | -8       |
| 10  | Opt_10   | 13   | 0.54   | 0.01   | 0.13  | 11      | -5       |
| ... | ...      | ...  | ...    | ...    | ...   | ...     | ...      |
| 120 | Opt_120  | 12   | 0.54   | 0.03   | 0.19  | 9       | -7       |

(Full data for all 120 options as retrieved above.)

#### Option_AssetReferenceMatrix.csv

For each $i=1,\dots,120$ and $j=1,\dots,6$, $A[i,j]$ is as in the retrieved data, e.g.:

| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3    | 0       | 0       | 0       | 0       | 0       | 1       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |
| Opt_120  | 0       | 1       | 0       | 0       | 1       | 0       |

#### Initial exposures and tolerances

- $\Delta_{\text{initial}} = 0.25$
- $\Gamma_{\text{initial}} = 0.08$
- $\text{Vega}_{\text{initial}} = 0.17$
- $\text{Tolerance}_\Delta = 0.06$
- $\text{Tolerance}_\Gamma = 0.05$
- $\text{Tolerance}_\text{Vega} = 0.07$

#### Complete Model

Minimize:
\[
\sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
\]
Subject to:
\[
\text{MaxShort}[i] \le x_i \le \text{MaxLong}[i] \qquad x_i \in \mathbb{Z} \qquad \forall i=1,\dots,120
\]
\[
y_i \ge x_i,\quad y_i \ge -x_i,\quad y_i \ge 0 \qquad \forall i=1,\dots,120
\]
\[
-0.06 \le 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \le 0.06
\]
\[
-0.05 \le 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \le 0.05
\]
\[
-0.07 \le 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \le 0.07
\]

All coefficients and bounds are as retrieved above, in source order.