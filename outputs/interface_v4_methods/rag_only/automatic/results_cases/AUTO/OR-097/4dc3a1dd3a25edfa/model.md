[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (across all 6 assets) are within specified risk tolerances after hedging. The solution must respect per-option trading limits and account for which assets each option references.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MIP) problem (due to integer variables and absolute values in the objective).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), to linearize the absolute value in the objective. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' (from OptionCharacteristics.csv), per option.
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' (from OptionCharacteristics.csv), per option.
        - Asset reference matrix \( A[i,j] \) (from Option_AssetReferenceMatrix.csv), binary, indicating which assets each option references.
    -   Constraint RHS (limits):
        - Initial net Delta, Gamma, Vega: 0.25, 0.08, 0.17 (given).
        - Risk band tolerances: |\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07 (given).
        - Trading limits: 'MaxLong', 'MaxShort' (from OptionCharacteristics.csv), per option.
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is implemented by minimizing \(\sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\), with constraints enforcing \(z[i] \geq x[i]\) and \(z[i] \geq -x[i]\).
7.  **Formulate Constraints:**
    -   **Absolute Value Linearization:** For each option \( i \):
        - \( z[i] \geq x[i] \)
        - \( z[i] \geq -x[i] \)
        - \( z[i] \geq 0 \)
    -   **Trading Limits:** For each option \( i \):
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \):
        - The net exposure after hedging is:
            \[
            G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]
            \]
          The absolute value of this sum must be within the specified tolerance:
            \[
            \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
            \]
          This is implemented as two linear inequalities for each Greek:
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \leq \text{Tolerance}_G \)
            - \( G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \geq -\text{Tolerance}_G \)
[Abstract Model Plan END]