[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to hedge a portfolio of 6 assets. The goal is to minimize total hedging cost while ensuring that, after hedging, the net Delta, Gamma, and Vega exposures of the portfolio are within specified tolerance bands. Each option’s per-contract cost, Greeks, and trading limits are given, and each option may reference one or more underlying assets as specified in a binary reference matrix.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer decision variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   \( x[i] \) = Integer number of contracts of option \( i \) to buy (positive) or sell (negative). Type: INTEGER.
    -   \( z[i] \) = Auxiliary variable representing \( |x[i]| \) for each option \( i \). Type: CONTINUOUS (or INTEGER, since \( x[i] \) is integer).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column from OptionCharacteristics.csv (per-contract cost for each option).
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' columns from OptionCharacteristics.csv (per-contract Greek exposures for each option).
    -   Asset reference: Option_AssetReferenceMatrix.csv, columns 'Asset_1' to 'Asset_6' (binary matrix \( A[i,j] \)).
    -   Trading limits: 'MaxLong' and 'MaxShort' columns from OptionCharacteristics.csv (upper and lower bounds for \( x[i] \)).
    -   Initial exposures: Provided as constants (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
    -   Risk band tolerances: Provided as constants (|\Delta| ≤ 0.06, |\Gamma| ≤ 0.05, |\text{Vega}| ≤ 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance:
        - For each Greek \( G \): \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
    -   **Trading Limits:** For each option \( i \): \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \)
    -   **Absolute Value Linking:** For each option \( i \): \( z[i] \geq x[i] \), \( z[i] \geq -x[i] \), and \( z[i] \geq 0 \) (to ensure \( z[i] = |x[i]| \))
[Abstract Model Plan END]