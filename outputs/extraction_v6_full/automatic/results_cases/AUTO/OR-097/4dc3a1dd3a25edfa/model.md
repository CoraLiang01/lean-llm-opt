[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of a 6-asset portfolio (after hedging) are within specified tolerance bands. Each option may reference one or more assets, and there are per-option trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute values in the objective and risk constraints, and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), to linearize the absolute value in the objective. Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Risk coefficients: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
    -   Asset reference: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary matrix A[i,j]).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerances: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (per-contract cost × absolute number of contracts):
    -   Minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute Value Linearization:** For each option \( i \):
        -   \( z[i] \geq x[i] \)
        -   \( z[i] \geq -x[i] \)
    -   **Trading Limits:** For each option \( i \):
        -   \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Risk Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \):
        -   Compute the net post-hedge exposure as:
            -   \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \)
        -   Impose the tolerance band:
            -   \( -\text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \leq \text{Tolerance}_G \)
        -   (Repeat for each Greek: Delta, Gamma, Vega, using their respective initial exposures and tolerances.)
    -   **Variable Types:** All \( x[i] \) are integer; all \( z[i] \) are integer (or continuous if allowed).
[Abstract Model Plan END]