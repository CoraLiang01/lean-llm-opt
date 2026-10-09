[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost for a portfolio of 6 assets. The solution must ensure that, after hedging, the net Delta, Gamma, and Vega exposures of the portfolio are within specified tolerance bands, while respecting per-option trading limits and accounting for which assets each option references.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MIP) problem with absolute value terms in the objective.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   \( x[i] \) = Integer number of contracts (positive for buy/long, negative for sell/short) for option \( i \). Type: GRB.INTEGER.
    -   \( z[i] \) = Auxiliary variable representing \( |x[i]| \) for option \( i \). Type: GRB.CONTINUOUS (used to linearize the absolute value in the objective).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' (from OptionCharacteristics.csv), per option.
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' (from OptionCharacteristics.csv), per option; 'A[i,j]' (from Option_AssetReferenceMatrix.csv), binary matrix indicating which assets each option references.
    -   Constraint RHS (limits): 'MaxLong', 'MaxShort' (from OptionCharacteristics.csv), per option; initial net Delta, Gamma, Vega exposures (given); risk band tolerances (given).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize the sum over all options of (Cost per contract) × (absolute value of contracts traded):  
    \(\min \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i]\), where \(z[i] = |x[i]|\).
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance:
        \[
        \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
        \]
        where \(G[i]\) is the per-contract Greek exposure for option \(i\), and \(A[i,j]\) indicates if option \(i\) references asset \(j\).
    -   **Trading Limits:** For each option \(i\), enforce:
        \[
        \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]
        \]
    -   **Absolute Value Linearization:** For each option \(i\), enforce:
        \[
        z[i] \geq x[i], \quad z[i] \geq -x[i], \quad z[i] \geq 0
        \]
[Abstract Model Plan END]