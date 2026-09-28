##### Sets and Indices

- $i = 1,\dots,120$: option index (corresponds to OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, source order)
- $j = 1,\dots,6$: asset index (corresponds to Asset_1, ..., Asset_6, source order)

##### Parameters (from OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, source order)

For $i=1,\dots,120$:
- $\text{Cost}[i]$: per-contract cost of option $i$
- $\Delta[i]$: per-contract delta of option $i$
- $\Gamma[i]$: per-contract gamma of option $i$
- $\text{Vega}[i]$: per-contract vega of option $i$
- $\text{MaxLong}[i]$: maximum allowed long position for option $i$
- $\text{MaxShort}[i]$: maximum allowed short position for option $i$ (negative integer)
- $A[i,j]$: 1 if option $i$ references asset $j$, 0 otherwise (from Option_AssetReferenceMatrix.csv, source order)

Initial exposures and tolerances:
- $\Delta_{\text{initial}} = 0.25$, $\Gamma_{\text{initial}} = 0.08$, $\text{Vega}_{\text{initial}} = 0.17$
- $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$

##### Decision Variables

- $x_i \in \mathbb{Z}$: integer number of contracts for option $i$ (positive for long, negative for short), $i=1,\dots,120$
- $y_i \geq 0$: auxiliary variable for $|x_i|$, $i=1,\dots,120$

##### Objective

\[
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
\]

##### Constraints

1. **Absolute value linking:**
   \[
   y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0,\quad \forall i=1,\dots,120
   \]

2. **Trading limits:**
   \[
   \text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i],\quad \forall i=1,\dots,120
   \]

3. **Delta risk band:**
   \[
   -\text{Tolerance}_\Delta \leq \Delta_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_\Delta
   \]

4. **Gamma risk band:**
   \[
   -\text{Tolerance}_\Gamma \leq \Gamma_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_\Gamma
   \]

5. **Vega risk band:**
   \[
   -\text{Tolerance}_\text{Vega} \leq \text{Vega}_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_\text{Vega}
   \]

6. **Variable domains:**
   \[
   x_i \in \mathbb{Z},\quad y_i \geq 0,\quad \forall i=1,\dots,120
   \]

##### Full Numerical Formulation

Let the options and assets be indexed in source order as in the CSVs:

- For $i=1,\dots,120$, let Option $i$ correspond to the $i$th row of OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv.
- For $j=1,\dots,6$, let Asset $j$ correspond to Asset_1, ..., Asset_6 in Option_AssetReferenceMatrix.csv.

\[
\begin{align*}
\min\ & \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i \\
\text{s.t.}\quad
& y_i \geq x_i,\quad \forall i=1,\dots,120 \\
& y_i \geq -x_i,\quad \forall i=1,\dots,120 \\
& y_i \geq 0,\quad \forall i=1,\dots,120 \\
& \text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i],\quad \forall i=1,\dots,120 \\
& -0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \leq 0.06 \\
& -0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \leq 0.05 \\
& -0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \leq 0.07 \\
& x_i \in \mathbb{Z},\quad y_i \geq 0,\quad \forall i=1,\dots,120
\end{align*}
\]

##### All coefficients and identifiers are as in the retrieved CSVs, in source order. For each $i$, use the values of Cost, Delta, Gamma, Vega, MaxLong, MaxShort from OptionCharacteristics.csv, and $A[i,j]$ from Option_AssetReferenceMatrix.csv.