##### Sets and Indices

- Let $I = \{1,2,\dots,120\}$ index the options, with identifiers $\text{Opt}_i$.
- Let $J = \{1,2,\dots,6\}$ index the assets, with identifiers $\text{Asset}_j$.

##### Parameters (from OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv)

For each option $i \in I$ (with identifier $\text{Opt}_i$):

- $\text{Cost}[i]$: per-contract cost
- $\Delta[i]$: per-contract delta
- $\Gamma[i]$: per-contract gamma
- $\text{Vega}[i]$: per-contract vega
- $\text{MaxLong}[i]$: maximum allowed long position (upper bound)
- $\text{MaxShort}[i]$: maximum allowed short position (lower bound)

For each option $i \in I$ and asset $j \in J$:

- $A[i,j]$: 1 if option $i$ references asset $j$, 0 otherwise

**All parameter values are as retrieved from the CSVs and preserved below.**

##### Initial Portfolio Exposures and Tolerances

- Initial net Delta: $\Delta_{\text{initial}} = 0.25$
- Initial net Gamma: $\Gamma_{\text{initial}} = 0.08$
- Initial net Vega: $\text{Vega}_{\text{initial}} = 0.17$
- Tolerances: $|\Delta| \leq 0.06$, $|\Gamma| \leq 0.05$, $|\text{Vega}| \leq 0.07$

##### Decision Variables

- $x_i \in \mathbb{Z}$: integer number of contracts for option $i$ ($i=1,\dots,120$), positive for long/buy, negative for short/sell
- $z_i \geq 0$: auxiliary variable for $|x_i|$ (for each $i$)

##### Objective Function

\[
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot z_i
\]

##### Constraints

1. **Absolute Value Linearization:**
   \[
   z_i \geq x_i,\quad z_i \geq -x_i,\quad z_i \geq 0,\quad \forall i=1,\dots,120
   \]

2. **Trading Limits:**
   \[
   \text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i],\quad \forall i=1,\dots,120
   \]

3. **Greek Exposure Constraints:**

   For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$, define $G[i]$ as the per-contract Greek for option $i$ and $G_{\text{initial}}$ as the initial exposure. Let $\text{Tolerance}_G$ be the risk band.

   \[
   \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \right| \leq \text{Tolerance}_G
   \]
   for $G = \Delta$ ($\text{Tolerance}_\Delta = 0.06$), $G = \Gamma$ ($\text{Tolerance}_\Gamma = 0.05$), $G = \text{Vega}$ ($\text{Tolerance}_\text{Vega} = 0.07$).

   These can be written as two linear inequalities for each Greek:
   \[
   -\text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_G
   \]

##### Full Data (as retrieved):

###### Option Characteristics

| Option   | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|----------|------|--------|-------|-------|---------|----------|
| Opt_1    | 9    | -0.54  | 0.12  | 0.1   | 9       | -14      |
| Opt_2    | 6    | 0.51   | 0.1   | 0.16  | 9       | -14      |
| Opt_3    | 13   | 0.17   | 0.02  | 0.19  | 10      | -7       |
| Opt_4    | 10   | -0.24  | 0.03  | 0.18  | 7       | -5       |
| Opt_5    | 7    | -0.61  | 0.14  | 0.11  | 12      | -9       |
| ...      | ...  | ...    | ...   | ...   | ...     | ...      |
| Opt_120  | 12   | 0.54   | 0.03  | 0.19  | 9       | -7       |

*(Full table includes all 120 options as in the CSV.)*

###### Option-Asset Reference Matrix

| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3    | 0       | 0       | 0       | 0       | 0       | 1       |
| Opt_4    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_5    | 0       | 0       | 1       | 0       | 0       | 0       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |
| Opt_120  | 0       | 1       | 0       | 0       | 1       | 0       |

*(Full matrix includes all 120 options and 6 assets as in the CSV.)*

##### Complete Mathematical Model

**Variables:**
- $x_i \in \mathbb{Z}$, $z_i \geq 0$ for $i=1,\dots,120$

**Objective:**
\[
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot z_i
\]

**Subject to:**
\[
\begin{align*}
& z_i \geq x_i,\quad z_i \geq -x_i,\quad z_i \geq 0,\quad \forall i=1,\dots,120 \\
& \text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i],\quad \forall i=1,\dots,120 \\
& -0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \leq 0.06 \\
& -0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \leq 0.05 \\
& -0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \leq 0.07 \\
& x_i \in \mathbb{Z},\quad \forall i=1,\dots,120
\end{align*}
\]

**Where all parameter values are as retrieved above from the CSVs.**