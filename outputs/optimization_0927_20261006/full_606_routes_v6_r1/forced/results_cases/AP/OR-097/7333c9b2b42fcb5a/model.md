##### Variables

Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $z_i \ge 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

##### Objective Function

$\min \sum_{i=1}^{120} \text{Cost}[i] \cdot z_i$

##### Constraints

###### 1. Absolute Value Constraints

For all $i=1,\dots,120$:
$$
z_i \ge x_i \\
z_i \ge -x_i
$$

###### 2. Trading Limits

For all $i=1,\dots,120$:
$$
\text{MaxShort}[i] \le x_i \le \text{MaxLong}[i]
$$

###### 3. Risk Exposure Constraints

Let $A[i,j]$ be the asset-reference matrix: $A[i,j]=1$ if option $i$ references asset $j$, $0$ otherwise, for $i=1,\dots,120$, $j=1,\dots,6$.

Let $\Delta[i]$, $\Gamma[i]$, $\text{Vega}[i]$ be the per-contract Greeks for option $i$.

Let $\Delta_{\text{initial}} = 0.25$, $\Gamma_{\text{initial}} = 0.08$, $\text{Vega}_{\text{initial}} = 0.17$.

Let $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$.

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
For all $j=1,\dots,6$ (assets):

$$
- \text{Tolerance}_G \le G_{\text{initial}} + \sum_{i=1}^{120} G[i] \cdot A[i,j] \cdot x_i \le \text{Tolerance}_G
$$

That is, for each asset $j$:
\[
\begin{align*}
-0.06 &\le 0.25 + \sum_{i=1}^{120} \Delta[i] \cdot A[i,j] \cdot x_i \le 0.06 \\
-0.05 &\le 0.08 + \sum_{i=1}^{120} \Gamma[i] \cdot A[i,j] \cdot x_i \le 0.05 \\
-0.07 &\le 0.17 + \sum_{i=1}^{120} \text{Vega}[i] \cdot A[i,j] \cdot x_i \le 0.07 \\
\end{align*}
\]

###### 4. Integrality

For all $i=1,\dots,120$:
$$
x_i \in \mathbb{Z}, \quad z_i \ge 0
$$

---

##### Retrieved Information

```json
{
  "OptionCharacteristics": [
    {"Option": "Opt_1", "Cost": 9, "Delta": -0.54, "Gamma": 0.12, "Vega": 0.1, "MaxLong": 9, "MaxShort": -14},
    {"Option": "Opt_2", "Cost": 6, "Delta": 0.51, "Gamma": 0.1, "Vega": 0.16, "MaxLong": 9, "MaxShort": -14},
    {"Option": "Opt_3", "Cost": 13, "Delta": 0.17, "Gamma": 0.02, "Vega": 0.19, "MaxLong": 10, "MaxShort": -7},
    {"Option": "Opt_4", "Cost": 10, "Delta": -0.24, "Gamma": 0.03, "Vega": 0.18, "MaxLong": 7, "MaxShort": -5},
    {"Option": "Opt_5", "Cost": 7, "Delta": -0.61, "Gamma": 0.14, "Vega": 0.11, "MaxLong": 12, "MaxShort": -9},
    {"Option": "Opt_6", "Cost": 9, "Delta": -0.26, "Gamma": 0.09, "Vega": 0.24, "MaxLong": 5, "MaxShort": -13},
    {"Option": "Opt_7", "Cost": 12, "Delta": -0.24, "Gamma": 0.01, "Vega": 0.2, "MaxLong": 10, "MaxShort": -5},
    {"Option": "Opt_8", "Cost": 5, "Delta": 0.32, "Gamma": 0.02, "Vega": 0.16, "MaxLong": 8, "MaxShort": -7},
    {"Option": "Opt_9", "Cost": 9, "Delta": 0.19, "Gamma": 0.1, "Vega": 0.17, "MaxLong": 5, "MaxShort": -8},
    {"Option": "Opt_10", "Cost": 13, "Delta": 0.54, "Gamma": 0.01, "Vega": 0.13, "MaxLong": 11, "MaxShort": -5},
    ...
    {"Option": "Opt_120", "Cost": 12, "Delta": 0.54, "Gamma": 0.03, "Vega": 0.19, "MaxLong": 9, "MaxShort": -7}
  ],
  "Option_AssetReferenceMatrix": [
    {"Option": "Opt_1", "Asset_1": 0, "Asset_2": 0, "Asset_3": 0, "Asset_4": 1, "Asset_5": 0, "Asset_6": 0},
    {"Option": "Opt_2", "Asset_1": 1, "Asset_2": 1, "Asset_3": 0, "Asset_4": 0, "Asset_5": 0, "Asset_6": 0},
    {"Option": "Opt_3", "Asset_1": 0, "Asset_2": 0, "Asset_3": 0, "Asset_4": 0, "Asset_5": 0, "Asset_6": 1},
    ...
    {"Option": "Opt_120", "Asset_1": 0, "Asset_2": 1, "Asset_3": 0, "Asset_4": 0, "Asset_5": 1, "Asset_6": 0}
  ],
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
```

- Option characteristics (Cost, Delta, Gamma, Vega, MaxLong, MaxShort) for all 120 options are provided.
- Asset reference matrix $A[i,j]$ for all 120 options and 6 assets is provided.
- Initial net Greeks and tolerances are given.

##### Indices

- Options: $i=1,\dots,120$ (Option names: Opt_1, ..., Opt_120)
- Assets: $j=1,\dots,6$ (Asset_1, ..., Asset_6)

##### Model Summary

Find integer $x_i$ for $i=1,\dots,120$ to minimize total hedging cost, subject to:
- Per-option trading limits,
- For each asset and each Greek, the net exposure after hedging is within the specified tolerance band,
- $z_i = |x_i|$ for cost calculation.