[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (across all 6 assets) are within specified tolerance bands after hedging. Each option may reference one or more assets, and each has its own per-contract cost, Greek exposures, and trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MIP) problem with absolute values in the objective (requiring auxiliary variables for linearization).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Per-contract cost: 'Cost' column in OptionCharacteristics.csv.
    -   Per-contract Greeks: 'Delta', 'Gamma', 'Vega' columns in OptionCharacteristics.csv.
    -   Option-asset mapping: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6', with A[i,j]=1 if option i references asset j.
    -   Trading limits: 'MaxLong', 'MaxShort' columns in OptionCharacteristics.csv.
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerance bands: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). Since absolute values are not linear, introduce auxiliary variables \(z[i]\) with constraints \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\), and minimize \(\sum_{i} \text{Cost}[i] \cdot z[i]\).
7.  **Formulate Constraints:**
    -   **Trading Limits:** For each option \( i \), enforce \(\text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]\), where MaxShort and MaxLong are from OptionCharacteristics.csv.
    -   **Auxiliary Variable Constraints:** For each option \( i \), enforce \(z[i] \geq x[i]\) and \(z[i] \geq -x[i]\), and \(z[i] \geq 0\).
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band. Specifically, for each Greek:
        - Compute the total net exposure as: \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \)
        - Enforce: \( |G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]| \leq \text{Tolerance}_G \)
        - This is a single constraint per Greek, aggregating over all options and all assets as specified in the query.
[Abstract Model Plan END]