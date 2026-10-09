[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to hedge a portfolio of 6 assets. The goal is to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures (after hedging) for the portfolio are within specified tolerances. Each option’s per-contract cost, Greeks, and trading limits are given, and each option may reference one or more assets (as specified in a binary matrix).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, an integer hedging optimization with absolute value in the objective and banded risk constraints).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of contracts of option \( i \) to hold (positive for long/buy, negative for short/sell). Type: GRB.INTEGER.
    -   `z[i]` = Auxiliary variable representing \(|x[i]|\) for each option \( i \), to linearize the absolute value in the objective. Type: GRB.INTEGER (or GRB.CONTINUOUS if allowed by solver).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' (from OptionCharacteristics.csv), per option.
    -   Constraint coefficients:
        - 'Delta', 'Gamma', 'Vega' (from OptionCharacteristics.csv), per option.
        - Asset reference matrix \( A[i,j] \) (from Option_AssetReferenceMatrix.csv), binary, indicating which assets each option references.
    -   Constraint RHS (limits):
        - Initial net Delta, Gamma, Vega: 0.25, 0.08, 0.17 (given).
        - Tolerances: Delta ≤ 0.06, Gamma ≤ 0.05, Vega ≤ 0.07 (given).
        - Trading limits: 'MaxLong', 'MaxShort' (from OptionCharacteristics.csv), per option.
6.  **Formulate Objective:** Minimize the total hedging cost, i.e., minimize \(\sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]|\). This is implemented by minimizing \(\sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\), with constraints to ensure \(z[i] \geq x[i]\) and \(z[i] \geq -x[i]\) for all \(i\).
7.  **Formulate Constraints:**
    -   **Risk Exposure Constraints (for each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \)):**
        - For each Greek, the net exposure after hedging must be within the specified tolerance band:
            - \(|G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i]| \leq \text{Tolerance}_G\)
        - This is a single constraint per Greek, aggregating over all options and all assets referenced by each option.
    -   **Trading Limits (per option):**
        - For each option \( i \): \(\text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]\)
    -   **Absolute Value Linearization (per option):**
        - For each option \( i \): \(z[i] \geq x[i]\), \(z[i] \geq -x[i]\)
[Abstract Model Plan END]