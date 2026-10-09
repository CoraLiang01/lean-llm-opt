[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and the per-contract Greek exposures are only counted for the assets each option references. Each option also has upper and lower trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute values in the objective (min-sum-cost), integer variables, and linear constraints with absolute value bands.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), to linearize the absolute value in the objective. Type: GRB.INTEGER (since \(x[i]\) is integer).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Greek exposures: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek for each option).
    -   Option-asset mapping: Option_AssetReferenceMatrix.csv, with A[i,j]=1 if option \(i\) references asset \(j\), 0 otherwise.
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \(x[i]\)).
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerance bands: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is implemented by introducing auxiliary variables \(z[i]\) with constraints \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\), and minimizing \(\sum_{i} \text{Cost}[i] \cdot z[i]\).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \(G\) (Delta, Gamma, Vega), require that the net exposure after hedging is within the specified band. For each Greek:
        - Compute the total Greek exposure as: \(G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]\).
        - Impose: \(|G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]| \leq \text{Tolerance}_G\).
        - This is a single constraint per Greek, aggregating over all options and all assets each option references, as specified in the query.
    -   **Trading Limits:** For each option \(i\), enforce: \(\text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]\).
    -   **Auxiliary Variable Constraints:** For each option \(i\), enforce \(z[i] \geq x[i]\) and \(z[i] \geq -x[i]\), so that \(z[i] = |x[i]|\).
[Abstract Model Plan END]