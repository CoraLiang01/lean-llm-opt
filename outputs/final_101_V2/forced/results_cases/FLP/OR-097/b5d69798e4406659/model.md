##### Decision Variables

Let $x_i$ be the integer number of contracts for option $i$, $i=1,\dots,120$ (positive for buy/long, negative for sell/short).

##### Parameters

- $\text{Cost}[i]$: Per-contract cost of option $i$ (from OptionCharacteristics.csv).
- $\Delta[i]$: Per-contract delta of option $i$.
- $\Gamma[i]$: Per-contract gamma of option $i$.
- $\text{Vega}[i]$: Per-contract vega of option $i$.
- $\text{MaxLong}[i]$: Maximum allowed long position for option $i$.
- $\text{MaxShort}[i]$: Maximum allowed short position for option $i$.
- $A[i,j]$: Asset-reference matrix, $A[i,j]=1$ if option $i$ references asset $j$, $0$ otherwise (from Option_AssetReferenceMatrix.csv), $i=1,\dots,120$, $j=1,\dots,6$.
- $G_{\text{initial}}$: Initial net exposure for Greek $G$:
  - $\Delta_{\text{initial}} = 0.25$
  - $\Gamma_{\text{initial}} = 0.08$
  - $\text{Vega}_{\text{initial}} = 0.17$
- $\text{Tolerance}_G$: Risk band for each Greek:
  - $\text{Tolerance}_\Delta = 0.06$
  - $\text{Tolerance}_\Gamma = 0.05$
  - $\text{Tolerance}_\text{Vega} = 0.07$

##### Objective Function

\[
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot |x_i|
\]

##### Constraints

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
\[
\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \right| \leq \text{Tolerance}_G
\]

That is, explicitly:
- Delta constraint:
  \[
  \left| 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \right| \leq 0.06
  \]
- Gamma constraint:
  \[
  \left| 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \right| \leq 0.05
  \]
- Vega constraint:
  \[
  \left| 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.07
  \]

Trading limits for each option:
\[
\text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i], \quad i=1,\dots,120
\]

$x_i$ are integer variables.

##### Data

All parameter vectors and matrices are as retrieved from OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, with full identifiers and values preserved for all 120 options and 6 assets.

- $\text{Cost}[i]$, $\Delta[i]$, $\Gamma[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ for $i=1,\dots,120$ as in OptionCharacteristics.csv.
- $A[i,j]$ for $i=1,\dots,120$, $j=1,\dots,6$ as in Option_AssetReferenceMatrix.csv.

##### Complete Mathematical Model

\[
\begin{align*}
\min_{x_i \in \mathbb{Z}} \quad & \sum_{i=1}^{120} \text{Cost}[i] \cdot |x_i| \\
\text{s.t.} \quad
& \left| 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \right| \leq 0.06 \\
& \left| 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \right| \leq 0.05 \\
& \left| 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \right| \leq 0.07 \\
& \text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i], \quad i=1,\dots,120 \\
\end{align*}
\]

where all parameter values and matrices are as retrieved above from the CSV files, with full option and asset identifiers and values preserved.