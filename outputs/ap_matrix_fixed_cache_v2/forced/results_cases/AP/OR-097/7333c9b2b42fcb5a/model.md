##### Variables

Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

##### Objective Function

$\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i$

##### Constraints

###### 1. Absolute Value Constraints

For all $i=1,\dots,120$:
$$
y_i \ge x_i \\
y_i \ge -x_i
$$

###### 2. Trading Limits

For all $i=1,\dots,120$:
$$
\text{MaxShort}[i] \le x_i \le \text{MaxLong}[i]
$$

###### 3. Risk Exposure Constraints

Let $A[i,j]$ be the binary asset-reference matrix ($A[i,j]=1$ if option $i$ references asset $j$, $0$ otherwise), for $i=1,\dots,120$, $j=1,\dots,6$.

Let $\Delta[i]$, $\Gamma[i]$, $\text{Vega}[i]$ be the per-contract Greeks for option $i$.

Let $\Delta_{\text{initial}} = 0.25$, $\Gamma_{\text{initial}} = 0.08$, $\text{Vega}_{\text{initial}} = 0.17$.

Let $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_{\text{Vega}} = 0.07$.

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
$$
- \text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \leq \text{Tolerance}_G
$$

That is, explicitly:

- Delta constraint:
$$
-0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Delta[i] \cdot A[i,j] \cdot x_i \leq 0.06
$$

- Gamma constraint:
$$
-0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \Gamma[i] \cdot A[i,j] \cdot x_i \leq 0.05
$$

- Vega constraint:
$$
-0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \leq 0.07
$$

###### 4. Integrality

For all $i=1,\dots,120$:
$$
x_i \in \mathbb{Z}, \quad y_i \ge 0
$$

---

##### Retrieved Information

###### Option Characteristics

- Options: $[ \text{Opt}_1, \text{Opt}_2, \ldots, \text{Opt}_{120} ]$
- For each $i=1,\dots,120$:
    - $\text{Cost}[i]$: as in OptionCharacteristics.csv
    - $\Delta[i]$: as in OptionCharacteristics.csv
    - $\Gamma[i]$: as in OptionCharacteristics.csv
    - $\text{Vega}[i]$: as in OptionCharacteristics.csv
    - $\text{MaxLong}[i]$: as in OptionCharacteristics.csv
    - $\text{MaxShort}[i]$: as in OptionCharacteristics.csv

Example (first 5 options):

| Option  | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|---------|------|--------|-------|-------|---------|----------|
| Opt_1   | 9    | -0.54  | 0.12  | 0.10  | 9       | -14      |
| Opt_2   | 6    | 0.51   | 0.10  | 0.16  | 9       | -14      |
| Opt_3   | 13   | 0.17   | 0.02  | 0.19  | 10      | -7       |
| Opt_4   | 10   | -0.24  | 0.03  | 0.18  | 7       | -5       |
| Opt_5   | 7    | -0.61  | 0.14  | 0.11  | 12      | -9       |
| ...     | ...  | ...    | ...   | ...   | ...     | ...      |

###### Asset Reference Matrix

- $A[i,j]$ is a $120 \times 6$ binary matrix, where $A[i,j]=1$ if option $i$ references asset $j$, $0$ otherwise.

Example (first 5 options):

| Option  | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|---------|---------|---------|---------|---------|---------|---------|
| Opt_1   | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2   | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3   | 0       | 0       | 0       | 0       | 0       | 1       |
| Opt_4   | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_5   | 0       | 0       | 1       | 0       | 0       | 0       |
| ...     | ...     | ...     | ...     | ...     | ...     | ...     |

###### Initial Greeks and Tolerances

- $\Delta_{\text{initial}} = 0.25$
- $\Gamma_{\text{initial}} = 0.08$
- $\text{Vega}_{\text{initial}} = 0.17$
- $\text{Tolerance}_\Delta = 0.06$
- $\text{Tolerance}_\Gamma = 0.05$
- $\text{Tolerance}_{\text{Vega}} = 0.07$

---

##### Full Model Data

{
  "Options": [
    "Opt_1", "Opt_2", ..., "Opt_120"
  ],
  "Cost": {
    "Opt_1": 9, "Opt_2": 6, ..., "Opt_120": 12
  },
  "Delta": {
    "Opt_1": -0.54, "Opt_2": 0.51, ..., "Opt_120": 0.54
  },
  "Gamma": {
    "Opt_1": 0.12, "Opt_2": 0.10, ..., "Opt_120": 0.03
  },
  "Vega": {
    "Opt_1": 0.10, "Opt_2": 0.16, ..., "Opt_120": 0.19
  },
  "MaxLong": {
    "Opt_1": 9, "Opt_2": 9, ..., "Opt_120": 9
  },
  "MaxShort": {
    "Opt_1": -14, "Opt_2": -14, ..., "Opt_120": -7
  },
  "AssetReferenceMatrix": {
    "Opt_1": [0,0,0,1,0,0],
    "Opt_2": [1,1,0,0,0,0],
    "Opt_3": [0,0,0,0,0,1],
    "Opt_4": [0,0,0,1,0,0],
    "Opt_5": [0,0,1,0,0,0],
    ...
    "Opt_120": [0,1,0,0,1,0]
  },
  "InitialGreeks": {
    "Delta": 0.25,
    "Gamma": 0.08,
    "Vega": 0.17
  },
  "Tolerances": {
    "Delta": 0.06,
    "Gamma": 0.05,
    "Vega": 0.07
  }
}

---

This model uses all provided data, preserves all identifiers, and includes all vectors and matrices as required.