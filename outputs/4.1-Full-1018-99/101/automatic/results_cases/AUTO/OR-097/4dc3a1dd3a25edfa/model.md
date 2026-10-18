[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of the portfolio (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and each has per-contract Greek exposures, costs, and trading limits. The solution must respect per-option trading limits and use the asset-reference matrix to aggregate Greeks across all referenced assets.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer decision variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   \( x[i] \) = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   \( y[i] \) = Auxiliary variable representing \( |x[i]| \) for each option \( i \). Type: GRB.INTEGER (for linearization of the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
    -   Asset reference: Option_AssetReferenceMatrix.csv, binary matrix \( A[i,j] \) indicating which assets each option references.
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided as constants for each Greek (\( G_{\text{initial}} \)).
    -   Tolerance bands: Provided as constants for each Greek (\( \text{Tolerance}_G \)).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (per-contract cost) times the absolute value of the position:
    -   \( \min \sum_{i=1}^{120} \text{Cost}[i] \cdot y[i] \), where \( y[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging (initial exposure plus the sum over all options and all referenced assets of per-contract Greek exposure times the number of contracts) is within the specified tolerance band:
        -   \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        -   (One constraint for each Greek, aggregating over all referenced assets for each option.)
    -   **Trading Limits:** For each option \( i \), enforce:
        -   \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Absolute Value Linearization:** For each option \( i \), enforce:
        -   \( y[i] \geq x[i] \)
        -   \( y[i] \geq -x[i] \)
        -   \( y[i] \geq 0 \)
    -   **Variable Domains:** \( x[i] \) and \( y[i] \) are integer variables for all \( i \).
[Abstract Model Plan END]