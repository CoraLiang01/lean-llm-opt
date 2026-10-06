[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the post-hedging portfolio exposures to Delta, Gamma, and Vega (across all 6 assets) are within specified tolerance bands. Each option may reference one or more assets, and each has per-contract Greeks, cost, and trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greeks for each option).
    -   Asset reference: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary matrix \( A[i,j] \), 1 if option \( i \) references asset \( j \)).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided as constants (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerances: Provided as constants (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is implemented by introducing auxiliary variables \( z[i] \geq |x[i]| \) and minimizing \(\sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\).
7.  **Formulate Constraints:**
    -   **Absolute Value Linking:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \), so that \( z[i] = |x[i]| \).
    -   **Trading Limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \).
    -   **Greek Exposure Constraints:** For each Greek \( G \) (Delta, Gamma, Vega), require that the net exposure after hedging is within the specified tolerance band. For each Greek:
        - Compute the total Greek exposure as: \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \).
        - Enforce: \( |G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]| \leq \text{Tolerance}_G \).
    -   **Variable Types:** All \( x[i] \) are integer; all \( z[i] \) are integer (or continuous if allowed).
[Abstract Model Plan END]