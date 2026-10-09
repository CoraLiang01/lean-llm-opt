[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and each option has per-contract Greek exposures, costs, and trading limits. The asset-reference matrix determines which assets each option affects.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and multi-asset, multi-Greek risk constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to buy (if positive) or sell (if negative). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \). Type: GRB.INTEGER (for linearizing the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv.
    -   Greek exposures per contract: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv.
    -   Trading limits: 'MaxLong', 'MaxShort' columns from OptionCharacteristics.csv.
    -   Asset reference: binary matrix \( A[i,j] \) from Option_AssetReferenceMatrix.csv (1 if option \( i \) references asset \( j \), 0 otherwise).
    -   Initial exposures: \( G_{\text{initial}} \) for each Greek (given in query).
    -   Tolerance bands: \( \text{Tolerance}_G \) for each Greek (given in query).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Absolute Value Linearization:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \) to ensure \( z[i] = |x[i]| \).
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified band:
        - \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        - This is a single constraint per Greek, aggregating over all options and all referenced assets as specified.
    -   **Trading Limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \), using the values from OptionCharacteristics.csv.
[Abstract Model Plan END]