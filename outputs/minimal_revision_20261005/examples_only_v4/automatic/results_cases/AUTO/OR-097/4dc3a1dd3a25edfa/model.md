[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the post-hedge portfolio exposures to Delta, Gamma, and Vega (across 6 assets) are within specified risk tolerances, and that all per-option trading limits are respected. The model must use the per-option cost and Greeks, as well as the mapping of which options reference which assets.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective (due to |x_i|), and linear constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing the absolute value of \( x[i] \) (i.e., \( z[i] = |x[i]| \)). Type: GRB.INTEGER (or GRB.CONTINUOUS if fractional contracts were allowed, but here integer).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Asset mapping: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (A[i,j], binary, 1 if option i references asset j).
    -   Constraint RHS (limits):
        - Initial net Greeks: Delta = 0.25, Gamma = 0.08, Vega = 0.17 (provided in query).
        - Risk band tolerances: |\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07 (provided in query).
        - Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per option).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (per-contract cost × absolute number of contracts traded):  
    \[
    \text{Minimize} \quad \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]
    \]
    where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute value linking:** For each option \( i \):
        - \( z[i] \geq x[i] \)
        - \( z[i] \geq -x[i] \)
        - (Ensures \( z[i] = |x[i]| \))
    -   **Trading limits:** For each option \( i \):
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Greek risk constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \):
        - Compute the total post-hedge exposure as:
            \[
            G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]
            \]
        - Impose the risk band:
            \[
            \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
            \]
        - This is implemented as two linear inequalities for each Greek:
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \leq \text{Tolerance}_G \)
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \geq -\text{Tolerance}_G \)
    -   **Variable types:** All \( x[i] \) are integer variables; all \( z[i] \) are integer and non-negative.
[Abstract Model Plan END]