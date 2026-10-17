[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of a 6-asset portfolio after hedging are within specified tolerance bands. Each option may reference one or more assets, and there are per-option trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute value terms in the objective.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   \( x[i] \): Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    -   \( y[i] \): Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (for linearization of absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures).
        - Asset reference matrix: Option_AssetReferenceMatrix.csv, with \( A[i,j] \) indicating if option \( i \) references asset \( j \).
    -   Constraint RHS:
        - Initial exposures: \( G_{\text{initial}} \) for each Greek (given in query).
        - Tolerances: \( \text{Tolerance}_G \) for each Greek (given in query).
        - Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (per option).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot y[i] \), where \( y[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance:
        - \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        - (This is one constraint per Greek, aggregating over all options and all referenced assets.)
    -   **Trading Limits:** For each option \( i \), enforce:
        - \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Absolute Value Linearization:** For each option \( i \), enforce:
        - \( y[i] \geq x[i] \)
        - \( y[i] \geq -x[i] \)
        - \( y[i] \geq 0 \)
[Abstract Model Plan END]