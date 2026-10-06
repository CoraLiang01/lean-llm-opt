[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (across all 6 assets) are within specified tolerance bands after hedging. Each option may reference one or more assets, and there are per-option trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute values in the objective and banded (absolute value) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed for absolute value modeling).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
    -   Asset mapping: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary matrix A[i,j]).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerance bands: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (per-contract cost × absolute number of contracts traded):  
    \[
    \text{Minimize} \quad \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]
    \]
    where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute Value Modeling:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \) so that \( z[i] = |x[i]| \).
    -   **Trading Limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \).
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), compute the net exposure after hedging as:
        \[
        G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]
        \]
        and require that its absolute value is within the specified tolerance:
        \[
        \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
        \]
        (This is a single constraint per Greek, aggregating over all options and all assets as specified.)
    -   **Variable Types:** All \( x[i] \) are integer; all \( z[i] \) are integer (or continuous if allowed).
[Abstract Model Plan END]