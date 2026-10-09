Let $x_i$ be the integer number of contracts (positive for long, negative for short) for option $i$ (Option), $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A_{ij}$ be the binary indicator that option $i$ references asset $j$ ($j=1,\dots,6$).

Let $\text{Cost}_i$, $\Delta_i$, $\Gamma_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$ be the per-option coefficients from OptionCharacteristics.csv.

Let $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$.

Let $\text{Tol}_\Delta = 0.06$, $\text{Tol}_\Gamma = 0.05$, $\text{Tol}_\text{Vega} = 0.07$.

Define:
- For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$, and for each asset $j=1,\dots,6$,
  $$
  G^{\text{total}}_j = G_{\text{init}} + \sum_{i=1}^{120} G_i \cdot A_{ij} \cdot x_i
  $$

The model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}_i \cdot y_i
$$

Subject to:

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$ and each asset $j=1,\dots,6$:
$$
- \text{Tol}_G \leq G_{\text{init}} + \sum_{i=1}^{120} G_i \cdot A_{ij} \cdot x_i \leq \text{Tol}_G
$$

For all $i=1,\dots,120$:
$$
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i
$$

For all $i=1,\dots,120$:
$$
y_i \geq x_i \\
y_i \geq -x_i \\
y_i \geq 0 \\
x_i \in \mathbb{Z}
$$

Where all coefficients and bounds are as follows (source order preserved):

#### OptionCharacteristics.csv

| Option   | Cost | Delta  | Gamma  | Vega  | MaxLong | MaxShort |
|----------|------|--------|--------|-------|---------|----------|
| Opt_1    | 9    | -0.54  | 0.12   | 0.1   | 9       | -14      |
| Opt_2    | 6    | 0.51   | 0.1    | 0.16  | 9       | -14      |
| Opt_3    | 13   | 0.17   | 0.02   | 0.19  | 10      | -7       |
| Opt_4    | 10   | -0.24  | 0.03   | 0.18  | 7       | -5       |
| Opt_5    | 7    | -0.61  | 0.14   | 0.11  | 12      | -9       |
| Opt_6    | 9    | -0.26  | 0.09   | 0.24  | 5       | -13      |
| ...      | ...  | ...    | ...    | ...   | ...     | ...      |
| Opt_120  | 12   | 0.54   | 0.03   | 0.19  | 9       | -7       |

#### Option_AssetReferenceMatrix.csv

| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3    | 0       | 0       | 0       | 0       | 0       | 1       |
| Opt_4    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_5    | 0       | 0       | 1       | 0       | 0       | 0       |
| Opt_6    | 1       | 0       | 0       | 0       | 0       | 0       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |
| Opt_120  | 0       | 1       | 0       | 0       | 1       | 0       |

#### Initial exposures and tolerances

- $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$
- $|\Delta| \leq 0.06$, $|\Gamma| \leq 0.05$, $|\text{Vega}| \leq 0.07$

#### Variable domains

- $x_i \in \mathbb{Z}$, $\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i$
- $y_i \geq |x_i|$

#### Summary

Minimize
$$
\sum_{i=1}^{120} \text{Cost}_i \cdot y_i
$$

Subject to, for each $j=1,\dots,6$:
\[
\begin{align*}
-0.06 &\leq 0.25 + \sum_{i=1}^{120} \Delta_i \cdot A_{ij} \cdot x_i \leq 0.06 \\
-0.05 &\leq 0.08 + \sum_{i=1}^{120} \Gamma_i \cdot A_{ij} \cdot x_i \leq 0.05 \\
-0.07 &\leq 0.17 + \sum_{i=1}^{120} \text{Vega}_i \cdot A_{ij} \cdot x_i \leq 0.07 \\
\end{align*}
\]

For all $i=1,\dots,120$:
\[
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i,\quad y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0,\quad x_i \in \mathbb{Z}
\]

All coefficients and bounds are as given in the retrieved data above, preserving source order and identifiers.