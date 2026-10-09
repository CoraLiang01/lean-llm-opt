[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost for a portfolio of 6 assets. The solution must ensure that, after hedging, the net Delta, Gamma, and Vega exposures of the portfolio are within specified tolerance bands, while respecting per-option trading limits and accounting for which assets each option references.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer variables, absolute value in objective, linear constraints).
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \dots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \dots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    -   \( x[i] \) = Integer number of contracts for option \( i \) (positive for buy/long, negative for sell/short). Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' (from OptionCharacteristics.csv), used as per-contract cost for each option.
    -   Constraint coefficients: 'Delta', 'Gamma', 'Vega' (from OptionCharacteristics.csv), giving per-contract Greek exposures for each option.
    -   Asset reference matrix: \( A[i,j] \) (from Option_AssetReferenceMatrix.csv), binary indicator if option \( i \) references asset \( j \).
    -   Trading limits: 'MaxLong', 'MaxShort' (from OptionCharacteristics.csv), giving upper and lower bounds for \( x[i] \).
    -   Initial exposures: \( G_{\text{initial}} \) for each Greek (given in query).
    -   Risk band tolerances: \( \text{Tolerance}_G \) for each Greek (given in query).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot |x[i]| \), where \( |x[i]| \) is the absolute value of the integer position in each option.
7.  **Formulate Constraints:**
    -   **Greek Exposure Constraints:** For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \), require that the net exposure after hedging is within the specified tolerance:
        \[
        \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
        \]
        (where \( G[i] \) is the per-contract Greek exposure for option \( i \), and \( A[i,j] \) indicates if option \( i \) references asset \( j \)).
    -   **Trading Limits:** For each option \( i \), enforce:
        \[
        \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i]
        \]
        (where MaxShort is negative and MaxLong is positive, as per schema).
[Abstract Model Plan END]