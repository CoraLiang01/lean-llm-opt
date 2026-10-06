[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified risk tolerances. Each option may reference one or more of 6 underlying assets, and each option has per-contract Greeks, cost, and trading limits. The asset-option relationships are given in a binary matrix.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, an integer hedging optimization with absolute value in the objective and multi-asset, multi-Greek constraints).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to hold (positive for long/buy, negative for short/sell). Type: GRB.INTEGER.
    -   `abs_x[i]` = Auxiliary variable representing the absolute value of \( x[i] \) (for modeling the cost). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Asset reference matrix: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary, indicating which assets each option references).
    -   Constraint RHS (limits):
        - Initial net exposures: Delta = 0.25, Gamma = 0.08, Vega = 0.17 (given).
        - Risk band tolerances: |\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07 (given).
        - Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per option).
6.  **Formulate Objective:** Minimize the total hedging cost, which is the sum over all options of the per-contract cost times the absolute value of the position:
    - Minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]| \)
    - (Model the absolute value using auxiliary variables: \( |x[i]| = abs\_x[i] \), with constraints \( abs\_x[i] \geq x[i] \), \( abs\_x[i] \geq -x[i] \))
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance:
        - For each Greek \( G \):
            - \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
            - Where:
                - \( G_{\text{initial}} \) is the initial net exposure for Greek \( G \) (given).
                - \( G[i] \) is the per-contract Greek exposure for option \( i \) (from OptionCharacteristics.csv).
                - \( A[i,j] \) is the binary indicator if option \( i \) references asset \( j \) (from Option_AssetReferenceMatrix.csv).
                - \( \text{Tolerance}_G \) is the risk band for Greek \( G \) (given).
    -   **Trading Limits:** For each option \( i \), the number of contracts must be within the allowed trading limits:
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
        - Where 'MaxLong' and 'MaxShort' are from OptionCharacteristics.csv.
    -   **Absolute Value Constraints:** For each option \( i \), enforce:
        - \( abs\_x[i] \geq x[i] \)
        - \( abs\_x[i] \geq -x[i] \)
        - \( abs\_x[i] \geq 0 \)
[Abstract Model Plan END]