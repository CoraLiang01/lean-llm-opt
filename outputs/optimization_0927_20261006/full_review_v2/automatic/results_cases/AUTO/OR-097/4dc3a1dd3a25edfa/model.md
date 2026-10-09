[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of a 6-asset portfolio after hedging are within specified tolerance bands. Each option may reference one or more assets, and each has per-contract Greeks, cost, and trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute value terms in the objective and risk constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv.
    -   Per-contract Greeks: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv.
    -   Option-asset mapping: \( A[i,j] \) from Option_AssetReferenceMatrix.csv (binary matrix).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv.
    -   Initial exposures: \( G_{\text{initial}} \) for each Greek (given in query).
    -   Risk tolerances: \( \text{Tolerance}_G \) for each Greek (given in query).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute Value Linking:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \), so that \( z[i] = |x[i]| \).
    -   **Trading Limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \).
    -   **Risk Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified band:
        - For each Greek \( G \), enforce:
          \[
          \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
          \]
          (i.e., for each Greek, the sum of initial exposure plus the total effect of all option positions across all referenced assets must be within the tolerance band.)
[Abstract Model Plan END]