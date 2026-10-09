[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer positions (number of contracts, long or short) in 120 European options to hedge a portfolio of 6 assets, minimizing total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures (after hedging) for the portfolio are within specified tolerance bands. Each option may reference one or more assets, and each has per-contract Greeks, cost, and trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer decision variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to hold (positive for long, negative for short). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (for modeling absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Asset reference matrix: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (A[i,j] = 1 if option i references asset j, 0 otherwise).
    -   Constraint RHS:
        - Initial net Greeks: Delta = 0.25, Gamma = 0.08, Vega = 0.17 (provided in query).
        - Tolerances: Delta ≤ 0.06, Gamma ≤ 0.05, Vega ≤ 0.07 (provided in query).
        - Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per option).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (per-contract cost × absolute value of contracts traded):  
    \(\min \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\), where \(z[i] = |x[i]|\).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance band:
        - For each Greek \( G \):  
          \(\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G\)
    -   **Trading Limits:** For each option \( i \):  
        \(\text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]\)
    -   **Absolute Value Linking:** For each option \( i \):  
        \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\), \(z[i] \geq 0\) (to ensure \(z[i] = |x[i]|\))
    -   **Variable Domains:**  
        \(x[i]\) integer, \(z[i]\) integer and non-negative, for all \(i\).
[Abstract Model Plan END]