##### Decision Variables

Let $x_i$ be the integer number of contracts for option $i$ ($i=1,\dots,120$), where $x_i > 0$ means buy/long, $x_i < 0$ means sell/short.

Let $z_i \geq 0$ be an auxiliary variable representing $|x_i|$ for each $i=1,\dots,120$.

##### Parameters

- $C_i$: Per-contract cost of option $i$ (from OptionCharacteristics.csv)
- $\Delta_i$: Per-contract delta of option $i$
- $\Gamma_i$: Per-contract gamma of option $i$
- $Vega_i$: Per-contract vega of option $i$
- $A_{i,j}$: 1 if option $i$ references asset $j$ ($j=1,\dots,6$), 0 otherwise (from Option_AssetReferenceMatrix.csv)
- $L_i$: MaxLong for option $i$ (upper bound)
- $S_i$: MaxShort for option $i$ (lower bound)
- Initial exposures: $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $Vega_{\text{init}} = 0.17$
- Tolerances: $T_\Delta = 0.06$, $T_\Gamma = 0.05$, $T_{Vega} = 0.07$

##### Objective Function

\[
\min \sum_{i=1}^{120} C_i \cdot z_i
\]

##### Constraints

1. **Auxiliary variable for absolute value:**
   \[
   z_i \geq x_i,\quad z_i \geq -x_i,\quad z_i \geq 0,\quad \forall i=1,\dots,120
   \]

2. **Trading limits:**
   \[
   S_i \leq x_i \leq L_i,\quad x_i \in \mathbb{Z},\quad \forall i=1,\dots,120
   \]

3. **Greek risk constraints (for each $G \in \{\Delta, \Gamma, Vega\}$):**
   \[
   \left| G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \right| \leq T_G
   \]
   That is, for each Greek $G$:
   \[
   -T_G \leq G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \leq T_G
   \]
   where $G_i$ is the per-contract value for that Greek for option $i$.

##### Data

###### Option Characteristics (OptionCharacteristics.csv, all 120 options, first 10 shown for brevity):

| Option   | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|----------|------|--------|-------|-------|---------|----------|
| Opt_1    | 9    | -0.54  | 0.12  | 0.10  | 9       | -14      |
| Opt_2    | 6    | 0.51   | 0.10  | 0.16  | 9       | -14      |
| Opt_3    | 13   | 0.17   | 0.02  | 0.19  | 10      | -7       |
| Opt_4    | 10   | -0.24  | 0.03  | 0.18  | 7       | -5       |
| Opt_5    | 7    | -0.61  | 0.14  | 0.11  | 12      | -9       |
| Opt_6    | 9    | -0.26  | 0.09  | 0.24  | 5       | -13      |
| Opt_7    | 12   | -0.24  | 0.01  | 0.20  | 10      | -5       |
| Opt_8    | 5    | 0.32   | 0.02  | 0.16  | 8       | -7       |
| Opt_9    | 9    | 0.19   | 0.10  | 0.17  | 5       | -8       |
| Opt_10   | 13   | 0.54   | 0.01  | 0.13  | 11      | -5       |
| ...      | ...  | ...    | ...   | ...   | ...     | ...      |

(Full data for all 120 options is included in the CSV.)

###### Option-Asset Reference Matrix (Option_AssetReferenceMatrix.csv, all 120 options × 6 assets, first 10 shown for brevity):

| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3    | 0       | 0       | 0       | 0       | 0       | 1       |
| Opt_4    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_5    | 0       | 0       | 1       | 0       | 0       | 0       |
| Opt_6    | 1       | 0       | 0       | 0       | 0       | 0       |
| Opt_7    | 0       | 0       | 0       | 0       | 1       | 1       |
| Opt_8    | 0       | 0       | 1       | 0       | 0       | 0       |
| Opt_9    | 0       | 0       | 0       | 1       | 0       | 1       |
| Opt_10   | 0       | 0       | 1       | 0       | 0       | 0       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |

(Full data for all 120 options × 6 assets is included in the CSV.)

##### Complete Mathematical Model

Let $I = \{1,\dots,120\}$ (option indices), $J = \{1,\dots,6\}$ (asset indices).

\[
\begin{align*}
\min_{x_i, z_i} \quad & \sum_{i=1}^{120} C_i \cdot z_i \\
\text{s.t.} \quad
& z_i \geq x_i,\quad z_i \geq -x_i,\quad z_i \geq 0,\quad \forall i \in I \\
& S_i \leq x_i \leq L_i,\quad x_i \in \mathbb{Z},\quad \forall i \in I \\
& -T_\Delta \leq \Delta_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta_i \cdot A_{i,j} \cdot x_i \leq T_\Delta \\
& -T_\Gamma \leq \Gamma_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma_i \cdot A_{i,j} \cdot x_i \leq T_\Gamma \\
& -T_{Vega} \leq Vega_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} Vega_i \cdot A_{i,j} \cdot x_i \leq T_{Vega}
\end{align*}
\]

Where all $C_i$, $\Delta_i$, $\Gamma_i$, $Vega_i$, $L_i$, $S_i$, $A_{i,j}$ are as given in the CSV files above, and the initial exposures and tolerances are as specified.

##### Retrieved Information

- Option characteristics for all 120 options (OptionCharacteristics.csv)
- Option-asset reference matrix for all 120 options × 6 assets (Option_AssetReferenceMatrix.csv)
- Initial exposures: $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $Vega_{\text{init}} = 0.17$
- Tolerances: $T_\Delta = 0.06$, $T_\Gamma = 0.05$, $T_{Vega} = 0.07$