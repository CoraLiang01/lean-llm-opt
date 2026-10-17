[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and the per-contract Greek exposures are only counted for the assets each option references. Each option has its own trading limits (max long/short).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (for linearization of the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   OptionCharacteristics.csv (all rows required):
        -   'Cost': Per-contract cost for each option \( i \).
        -   'Delta', 'Gamma', 'Vega': Per-contract Greek exposures for each option \( i \).
        -   'MaxLong', 'MaxShort': Trading limits for each option \( i \).
    -   Option_AssetReferenceMatrix.csv (all rows required):
        -   'A[i,j]': Binary indicator if option \( i \) references asset \( j \).
    -   Initial exposures and tolerances (from query):
        -   \( G_{\text{initial}} \): Initial net Delta = 0.25, Gamma = 0.08, Vega = 0.17.
        -   \( \text{Tolerance}_\Delta = 0.06 \), \( \text{Tolerance}_\Gamma = 0.05 \), \( \text{Tolerance}_\text{Vega} = 0.07 \).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]| \). (Linearized using auxiliary variables \( z[i] \) with constraints \( z[i] \geq x[i] \), \( z[i] \geq -x[i] \).)
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band:
        -   \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        -   (For each Greek, sum over all options and all assets referenced by each option.)
    -   **Trading Limits:** For each option \( i \), enforce:
        -   \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Absolute Value Linearization:** For each option \( i \), enforce:
        -   \( z[i] \geq x[i] \)
        -   \( z[i] \geq -x[i] \)
        -   \( z[i] \geq 0 \)
    -   **Variable Domains:** All \( x[i] \) and \( z[i] \) are integer variables; \( z[i] \geq 0 \).
[Abstract Model Plan END]