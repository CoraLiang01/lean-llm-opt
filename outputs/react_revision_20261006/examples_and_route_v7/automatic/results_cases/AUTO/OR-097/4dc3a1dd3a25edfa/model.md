## Mathematical Model

Let:
- $I$ = set of options (from OptionCharacteristics.csv, column Option)
- $J$ = set of assets (from Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6)
- $x_i \in \mathbb{Z}$: integer number of contracts for option $i \in I$ (positive for long, negative for short)
- $c_i$: per-contract cost of option $i$ (file_0, Cost)
- $d_i$: per-contract Delta of option $i$ (file_0, Delta)
- $g_i$: per-contract Gamma of option $i$ (file_0, Gamma)
- $v_i$: per-contract Vega of option $i$ (file_0, Vega)
- $A_{ij}$: 1 if option $i$ references asset $j$, 0 otherwise (file_1, Asset_k)
- $L_i$, $S_i$: MaxLong and MaxShort for option $i$ (file_0, MaxLong, MaxShort)
- $G_{\text{init}}$: initial net Greek exposure for $G \in \{\Delta, \Gamma, \text{Vega}\}$ (given)
- $\text{Tol}_G$: risk band tolerance for $G$ (given)

### Variables

For all $i \in I$:
- $x_i \in \mathbb{Z}$, $S_i \le x_i \le L_i$
- $y_i \ge 0$, $y_i \ge x_i$, $y_i \ge -x_i$ (to model $|x_i|$)

### Objective

Minimize total hedging cost:
$$
\min \sum_{i \in I} c_i \cdot y_i
$$

### Constraints

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
- For all $j \in J$ (assets):
$$
\left| G_{\text{init}} + \sum_{i \in I} G_i \cdot A_{ij} \cdot x_i \right| \leq \text{Tol}_G
$$
where $G_i$ is $d_i$, $g_i$, or $v_i$ for Delta, Gamma, or Vega, respectively.

For all $i \in I$:
$$
S_i \le x_i \le L_i
$$
$$
y_i \ge x_i
$$
$$
y_i \ge -x_i
$$
$$
y_i \ge 0
$$

### Data Mapping

- $I$: file_0_view_0, Option
- $J$: file_1_view_0, Asset_1, ..., Asset_6 (column names)
- $c_i$: file_0_view_0, Cost
- $d_i$: file_0_view_0, Delta
- $g_i$: file_0_view_0, Gamma
- $v_i$: file_0_view_0, Vega
- $A_{ij}$: file_1_view_0, Option (row, matches Option in file_0), Asset_k (columns Asset_1,...,Asset_6)
- $L_i$: file_0_view_0, MaxLong
- $S_i$: file_0_view_0, MaxShort
- $G_{\text{init}}$: Delta: 0.25, Gamma: 0.08, Vega: 0.17 (given)
- $\text{Tol}_G$: Delta: 0.06, Gamma: 0.05, Vega: 0.07 (given)

All index sets, parameters, and constraints are mapped directly to the supplied data tables and columns. No data is omitted or synthesized.