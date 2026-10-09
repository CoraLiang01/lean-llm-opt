[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of a 6-asset portfolio (after hedging) are within specified tolerance bands. Each option may reference one or more assets, and there are per-option trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute values in the objective and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (for linearizing the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Greek exposures: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek for each option).
    -   Asset references: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary matrix \( A[i,j] \)).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerances: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is linearized by introducing auxiliary variables \( z[i] \geq |x[i]| \) for each \( i \).
7.  **Formulate Constraints:**
    -   **Absolute Value Linearization:** For each option \( i \):
        - \( z[i] \geq x[i] \)
        - \( z[i] \geq -x[i] \)
    -   **Trading Limits:** For each option \( i \):
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \):
        - For the entire portfolio (across all assets), the net exposure after hedging must satisfy:
            - \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        - This is implemented as two linear inequalities for each Greek:
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \leq \text{Tolerance}_G \)
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \geq -\text{Tolerance}_G \)
    -   **Variable Domains:** All \( x[i] \) and \( z[i] \) are integer variables, with \( z[i] \geq 0 \).
[Abstract Model Plan END]