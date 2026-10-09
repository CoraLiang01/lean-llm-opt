[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and the per-contract Greek exposures are only counted for the assets each option references. Each option also has upper and lower trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MIP) problem with absolute values in the objective (requiring auxiliary variables or equivalent reformulation).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv, all rows required)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv, all columns Asset_1 to Asset_6)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \) (fixed set, as specified in the query)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Greek exposures: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek for each option).
    -   Option-asset mapping: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6', with A[i,j]=1 if option i references asset j, 0 otherwise.
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided in the query (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Tolerance bands: Provided in the query (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). Since absolute values are involved, introduce auxiliary variables \( z[i] \geq |x[i]| \) for each \( i \), and minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\).
7.  **Formulate Constraints:**
    -   **Absolute value linking:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \), so that \( z[i] = |x[i]| \).
    -   **Trading limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \), using the values from OptionCharacteristics.csv.
    -   **Greek exposure constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band. Specifically:
        - Compute the total Greek exposure added by the options as follows:
            - For each Greek \( G \), sum over all options and all assets: \( \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \)
            - Add the initial exposure \( G_{\text{initial}} \)
            - Impose the constraint: \( | G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] | \leq \text{Tolerance}_G \)
        - For each Greek, this is a single constraint (not per-asset), aggregating all referenced assets and options as specified.
    -   **Variable domains:** \( x[i] \) are integer variables; \( z[i] \) are continuous and non-negative.
[Abstract Model Plan END]