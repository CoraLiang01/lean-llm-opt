[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net portfolio exposures to Delta, Gamma, and Vega (after hedging) are within specified tolerance bands. Each option may reference one or more of 6 underlying assets, and the per-contract Greek exposures are only counted for the assets each option references. Each option has its own trading limits (max long and max short).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with absolute value terms in the objective and integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    - \( x[i] \): Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
    - \( z[i] \): Auxiliary variable representing \( |x[i]| \) for each option \( i \). Type: GRB.INTEGER (to linearize the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    - From OptionCharacteristics.csv (all 120 rows required, no filter):
        - 'Cost': Per-contract cost for each option \( i \).
        - 'Delta', 'Gamma', 'Vega': Per-contract Greek exposures for each option \( i \).
        - 'MaxLong', 'MaxShort': Trading limits for each option \( i \).
    - From Option_AssetReferenceMatrix.csv (all 120 rows required, no filter):
        - \( A[i,j] \): Binary indicator if option \( i \) references asset \( j \).
    - Initial exposures and tolerances (from query, not CSV):
        - \( G_{\text{initial}} \): Initial net exposure for each Greek (\( \Delta = 0.25, \Gamma = 0.08, \text{Vega} = 0.17 \)).
        - \( \text{Tolerance}_G \): Risk band for each Greek (\( |\Delta| \leq 0.06, |\Gamma| \leq 0.05, |\text{Vega}| \leq 0.07 \)).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    - **Absolute Value Linearization:** For each option \( i \), enforce \( z[i] \geq x[i] \) and \( z[i] \geq -x[i] \), so \( z[i] = |x[i]| \).
    - **Trading Limits:** For each option \( i \), enforce \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \).
    - **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), enforce:
        - \( \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G \)
        - That is, the net exposure after hedging (initial plus sum over all options and referenced assets) must be within the specified band for each Greek.
[Abstract Model Plan END]