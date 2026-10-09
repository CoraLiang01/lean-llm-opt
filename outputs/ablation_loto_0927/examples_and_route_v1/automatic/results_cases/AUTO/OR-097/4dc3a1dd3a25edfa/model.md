Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $C_i$, $\Delta_i$, $\Gamma_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$ be the cost, delta, gamma, vega, max long, and max short for option $i$ (from OptionCharacteristics.csv).

Let $A_{i,j}$ be the binary indicator that option $i$ references asset $j$ (from Option_AssetReferenceMatrix.csv), $j=1,\dots,6$.

Let the initial net exposures be:
- $\Delta_{\text{initial}} = 0.25$
- $\Gamma_{\text{initial}} = 0.08$
- $\text{Vega}_{\text{initial}} = 0.17$

Let the risk tolerances be:
- $|\Delta| \le 0.06$
- $|\Gamma| \le 0.05$
- $|\text{Vega}| \le 0.07$

The complete model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} C_i \cdot y_i
$$

Subject to:

For all $i=1,\dots,120$:
\[
\begin{align*}
& y_i \ge x_i \\
& y_i \ge -x_i \\
& \text{MaxShort}_i \le x_i \le \text{MaxLong}_i \\
& x_i \in \mathbb{Z}
\end{align*}
\]

Risk constraints (for each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$):

Let $G_{\text{initial}}$ be the initial exposure for Greek $G$.

Let $G_i$ be the per-contract Greek exposure for option $i$.

Let $\text{Tolerance}_G$ be the risk band for Greek $G$.

For each Greek $G$:
\[
\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_G
\]

Explicitly, for the three Greeks:

Delta:
\[
-0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta_i \cdot A_{i,j} \cdot x_i \leq 0.06
\]

Gamma:
\[
-0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma_i \cdot A_{i,j} \cdot x_i \leq 0.05
\]

Vega:
\[
-0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}_i \cdot A_{i,j} \cdot x_i \leq 0.07
\]

Where all coefficients $C_i$, $\Delta_i$, $\Gamma_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$ and $A_{i,j}$ are taken directly from the retrieved OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, preserving their original order and identifiers.

All variables $x_i$ are integer, and $y_i$ are continuous and nonnegative.

This model uses all retrieved data and enforces all constraints as described.