Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A[i,j]$ be the binary indicator that option $i$ references asset $j$, for $i=1,\dots,120$, $j=1,\dots,6$ (from Option_AssetReferenceMatrix.csv).

Let $\text{Cost}[i]$, $\text{Delta}[i]$, $\text{Gamma}[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ be the per-option coefficients from OptionCharacteristics.csv.

Constants:
- Initial net Delta: $0.25$
- Initial net Gamma: $0.08$
- Initial net Vega: $0.17$
- Tolerances: $|\Delta| \le 0.06$, $|\Gamma| \le 0.05$, $|\text{Vega}| \le 0.07$

Define for each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
- $G_{\text{initial}}$ is the initial exposure for that Greek.
- $G[i]$ is the per-contract Greek exposure of option $i$.

The model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
$$

Subject to:

For each $i=1,\dots,120$:
\[
\begin{align*}
& y_i \ge x_i \\
& y_i \ge -x_i \\
& \text{MaxShort}[i] \le x_i \le \text{MaxLong}[i] \\
& x_i \in \mathbb{Z}
\end{align*}
\]

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
\[
\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \right| \leq \text{Tolerance}_G
\]

Explicitly, using the provided data:

- For Delta:
  \[
  \left| 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Delta}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.06
  \]
- For Gamma:
  \[
  \left| 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Gamma}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.05
  \]
- For Vega:
  \[
  \left| 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.07
  \]

Where all coefficients and bounds are taken directly from the retrieved OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, preserving their order and identifiers.

All variables $x_i$ are integer, $y_i$ are continuous and nonnegative.

All constraints and coefficients are as above, with no omitted data or invented values.