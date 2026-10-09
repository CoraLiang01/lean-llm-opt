[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and the per-contract Greek exposures are only counted for the assets referenced by each option. Each option also has individual trading limits (max long and max short).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective (can be linearized), and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv, all rows required)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv, all rows and columns required)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \) (explicitly enumerated in the query)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER, for \( i = 1, \ldots, 120 \).
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), used to linearize the absolute value in the objective. Type: GRB.INTEGER, for \( i = 1, \ldots, 120 \).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Greek exposures: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek for each option).
    -   Option-asset mapping: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6', binary matrix \( A[i,j] \) indicating which assets each option references.
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per-option upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided in the query as \( \Delta_{\text{initial}} = 0.25 \), \( \Gamma_{\text{initial}} = 0.08 \), \( \text{Vega}_{\text{initial}} = 0.17 \).
    -   Tolerances: Provided in the query as \( |\Delta| \leq 0.06 \), \( |\Gamma| \leq 0.05 \), \( |\text{Vega}| \leq 0.07 \).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of per-contract cost times the absolute value of the position:
    -   Objective: Minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute Value Linearization:** For each option \( i \):
        - \( z[i] \geq x[i] \)
        - \( z[i] \geq -x[i] \)
    -   **Trading Limits:** For each option \( i \):
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band. For each Greek:
        - Compute the total net exposure as:
            - \( G_{\text{net}} = G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \)
        - Impose the band:
            - \( -\text{Tolerance}_G \leq G_{\text{net}} \leq \text{Tolerance}_G \)
        - This is a single constraint per Greek, aggregating over all options and all assets referenced by each option, as specified in the query.
[Abstract Model Plan END]