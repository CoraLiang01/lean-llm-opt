[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the post-hedging portfolio exposures to Delta, Gamma, and Vega (across 6 assets) are within specified tolerance bands. Each option may reference one or more assets, and each has per-contract Greeks, cost, and trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective (min-sum-cost), integer variables, and linear constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), to linearize the absolute value in the objective. Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed by solver).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (cost per contract for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Asset reference matrix: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (A[i,j], binary).
    -   Constraint RHS (limits):
        - Initial net Greeks: Delta = 0.25, Gamma = 0.08, Vega = 0.17 (provided in query).
        - Tolerances: Delta ≤ 0.06, Gamma ≤ 0.05, Vega ≤ 0.07 (provided in query).
        - Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per option).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is implemented by introducing auxiliary variables \(z[i]\) with constraints \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\), and minimizing \(\sum_{i} \text{Cost}[i] \cdot z[i]\).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints (for each Greek \( G \)):**
        - For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require:
            \[
            \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
            \]
        - This is implemented as two inequalities per Greek:
            \[
            G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \leq \text{Tolerance}_G
            \]
            \[
            G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \geq -\text{Tolerance}_G
            \]
    -   **Trading Limits (per option):**
        - For each option \( i \):
            \[
            \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]
            \]
    -   **Auxiliary Variable Constraints (for absolute value):**
        - For each option \( i \):
            \[
            z[i] \geq x[i]
            \]
            \[
            z[i] \geq -x[i]
            \]
            \[
            z[i] \geq 0
            \]
[Abstract Model Plan END]