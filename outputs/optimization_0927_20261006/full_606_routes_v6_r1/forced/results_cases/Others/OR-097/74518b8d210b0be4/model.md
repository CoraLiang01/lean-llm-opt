[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of contracts to buy or sell for each of 120 European options, in order to minimize total hedging cost, while ensuring that the net Delta, Gamma, and Vega exposures of a 6-asset portfolio after hedging are within specified tolerance bands. Each option may reference one or more assets, and there are per-option trading limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) problem with absolute value terms in the objective.
3.  **Define Index Sets:** The primary indices are:
    - Options: \( i = 1, \ldots, 120 \) (from OptionCharacteristics.csv)
    - Assets: \( j = 1, \ldots, 6 \) (from Option_AssetReferenceMatrix.csv)
    - Greeks: \( G \in \{\Delta, \Gamma, \text{Vega}\} \)
4.  **Define Decision Variables:**
    - \( x[i] \): Integer number of contracts of option \( i \) to buy (positive) or sell (negative). Type: GRB.INTEGER.
    - \( z[i] \): Auxiliary variable representing \( |x[i]| \) for each option \( i \). Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    - OptionCharacteristics.csv:
        - 'Cost': Per-contract cost for each option (\( \text{Cost}[i] \)), used in the objective.
        - 'Delta', 'Gamma', 'Vega': Per-contract Greek exposures (\( \text{Delta}[i] \), \( \text{Gamma}[i] \), \( \text{Vega}[i] \)), used in constraints.
        - 'MaxLong', 'MaxShort': Upper and lower trading limits for each option (\( \text{MaxLong}[i] \), \( \text{MaxShort}[i] \)), used as variable bounds.
    - Option_AssetReferenceMatrix.csv:
        - \( A[i,j] \): Binary indicator if option \( i \) references asset \( j \).
    - Initial exposures and tolerances (from query):
        - \( G_{\text{initial}} \): Initial net exposure for each Greek (Delta = 0.25, Gamma = 0.08, Vega = 0.17).
        - \( \text{Tolerance}_G \): Risk band for each Greek (Delta: 0.06, Gamma: 0.05, Vega: 0.07).
6.  **Formulate Objective:** Minimize total hedging cost, i.e., minimize \( \sum_{i=1}^{120} \text{Cost}[i] \cdot z[i] \), where \( z[i] = |x[i]| \).
7.  **Formulate Constraints:**
    - For each Greek \( G \in \{\Delta, \Gamma, \text{Vega}\} \):
        - The net post-hedge exposure must satisfy:
            \[
            \left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x[i] \right| \leq \text{Tolerance}_G
            \]
          (where \( G[i] \) is the per-contract Greek for option \( i \), and \( A[i,j] \) indicates asset reference).
    - For each option \( i \):
        - Trading limits: \( \text{MaxShort}[i] \leq x[i] \leq \text{MaxLong}[i] \).
        - Absolute value definition: \( z[i] \geq x[i] \), \( z[i] \geq -x[i] \), \( z[i] \geq 0 \).
[Abstract Model Plan END]