[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost for a portfolio of 6 assets. The solution must ensure that, after hedging, the net Delta, Gamma, and Vega exposures of the portfolio are within specified tolerance bands. Each option’s per-contract cost, Greeks, and trading limits are given, and each option may reference one or more assets (as specified in a binary matrix).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, an integer hedging optimization with absolute value in the objective and multi-asset, multi-Greek constraints).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed for absolute value modeling).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Option_AssetReferenceMatrix.csv: binary matrix A[i,j] indicating which asset(s) each option references.
    -   Constraint RHS (limits):
        - Initial net Delta, Gamma, Vega exposures: 0.25, 0.08, 0.17 (given).
        - Tolerance bands: |\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07 (given).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per-option upper and lower bounds for x[i]).
6.  **Formulate Objective:** Minimize total hedging cost, defined as the sum over all options of per-contract cost times the absolute value of the position:
    - Minimize: \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\)
    - (Model absolute value using auxiliary variables: \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\), \(z[i] \geq 0\), and minimize \(\sum \text{Cost}[i] \cdot z[i]\))
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band:
        - For each Greek \( G \):
            - \(\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G\)
        - (Model absolute value as two inequalities for each Greek.)
    -   **Trading Limits:** For each option \( i \):
        - \(\text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]\)
    -   **Auxiliary Variable Constraints:** For each option \( i \):
        - \(z[i] \geq x[i]\)
        - \(z[i] \geq -x[i]\)
        - \(z[i] \geq 0\)
[Abstract Model Plan END]