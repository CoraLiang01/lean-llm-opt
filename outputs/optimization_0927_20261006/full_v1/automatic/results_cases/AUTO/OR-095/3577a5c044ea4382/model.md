[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production quantities for each of 141 aerospace-grade widgets (Widget1–Widget141), considering labor, material, and byproduct constraints, to maximize total profit. Special attention is required for Widget3, which generates a valuable byproduct (CatalystX) that can be sold up to a capped amount, with disposal costs for any excess.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) resource allocation and byproduct management problem.
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \{\text{Widget1}, \ldots, \text{Widget141}\} \)
4.  **Define Decision Variables:**
    - \( x[i] \) = Number of units of widget \( i \) to produce in the month. Type: GRB.CONTINUOUS.
    - \( s \) = Amount (kg) of CatalystX sold in the month. Type: GRB.CONTINUOUS.
    - \( d \) = Amount (kg) of CatalystX disposed of in the month. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'Profit' (from product_resources.csv) for each widget \( i \).
        - Sale price of CatalystX: $300/kg (external parameter).
        - Disposal cost of CatalystX: $200/kg (external parameter).
    - Constraint coefficients:
        - 'LaborHours', 'MaterialA', 'MaterialB' (from product_resources.csv) per unit of widget \( i \).
        - CatalystX generation rate: 5 kg per unit of Widget3 (external parameter).
    - Constraint RHS (limits):
        - 'MonthlyLimit' for 'LaborHours', 'MaterialA', 'MaterialB' (from resource_limits.csv).
        - CatalystX sales cap: 1500 kg/month (external parameter).
6.  **Formulate Objective:** Maximize total profit, which is the sum of:
    - Total base profit from all widgets produced: \( \sum_{i} \text{Profit}[i] \times x[i] \)
    - Plus revenue from CatalystX sold: \( 300 \times s \)
    - Minus disposal cost for unsold CatalystX: \( 200 \times d \)
    - So, maximize: \( \sum_{i} \text{Profit}[i] \times x[i] + 300 \times s - 200 \times d \)
7.  **Formulate Constraints:**
    - Resource Constraints:
        - Labor: \( \sum_{i} \text{LaborHours}[i] \times x[i] \leq \text{MonthlyLimit}[\text{LaborHours}] \)
        - Material A: \( \sum_{i} \text{MaterialA}[i] \times x[i] \leq \text{MonthlyLimit}[\text{MaterialA}] \)
        - Material B: \( \sum_{i} \text{MaterialB}[i] \times x[i] \leq \text{MonthlyLimit}[\text{MaterialB}] \)
    - CatalystX Byproduct Balance:
        - Total CatalystX generated: \( 5 \times x[\text{Widget3}] = s + d \)
    - CatalystX Sales Cap:
        - \( s \leq 1500 \)
    - Non-negativity:
        - \( x[i] \geq 0 \) for all widgets \( i \)
        - \( s \geq 0 \)
        - \( d \geq 0 \)
[Abstract Model Plan END]