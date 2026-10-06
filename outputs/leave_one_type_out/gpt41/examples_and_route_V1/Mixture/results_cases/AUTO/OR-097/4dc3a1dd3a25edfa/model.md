Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A[i,j]$ be the binary indicator from Option_AssetReferenceMatrix.csv: $A[i,j]=1$ if option $i$ references asset $j$, $0$ otherwise, for $i=1,\dots,120$, $j=1,\dots,6$.

Let $\text{Cost}[i]$, $\text{Delta}[i]$, $\text{Gamma}[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ be the coefficients from OptionCharacteristics.csv for $i=1,\dots,120$.

Constants:
- Initial net Delta: $0.25$
- Initial net Gamma: $0.08$
- Initial net Vega: $0.17$
- Tolerances: $|\Delta| \le 0.06$, $|\Gamma| \le 0.05$, $|\text{Vega}| \le 0.07$

The complete model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
$$

Subject to:

For each $i=1,\dots,120$:
\begin{align*}
    \text{MaxShort}[i] &\le x_i \le \text{MaxLong}[i] \\
    y_i &\ge x_i \\
    y_i &\ge -x_i \\
    y_i &\ge 0 \\
    x_i &\in \mathbb{Z}
\end{align*}

Risk constraints (using the asset-reference matrix):

Let $S_G = G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i$ for $G \in \{\text{Delta}, \text{Gamma}, \text{Vega}\}$.

\[
\begin{align*}
\left| 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Delta}[i] \cdot A[i,j] \cdot x_i \right| &\le 0.06 \\
\left| 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Gamma}[i] \cdot A[i,j] \cdot x_i \right| &\le 0.05 \\
\left| 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \right| &\le 0.07 \\
\end{align*}
\]

Or, equivalently, for each Greek $G$:
\[
- \text{Tolerance}_G \le G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \le \text{Tolerance}_G
\]

Where all coefficients and bounds are taken directly from the retrieved CSVs, preserving their order and identifiers.

All variables $x_i$ are integer, $y_i$ are continuous and nonnegative.

All data used is as retrieved above, with no omitted rows or synthesized identifiers.