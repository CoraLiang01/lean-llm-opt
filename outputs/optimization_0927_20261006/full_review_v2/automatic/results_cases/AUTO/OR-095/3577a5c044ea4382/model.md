[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production quantities for each of 141 aerospace-grade widgets (Widget1–Widget141), considering labor, material, and byproduct constraints, to maximize total profit. Special attention is required for Widget3, which generates a valuable byproduct (CatalystX) that can be sold (with a sales cap) or must be disposed of at a cost if unsold.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) resource allocation and byproduct management problem.
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \{\text{Widget1}, \ldots, \text{Widget141}\} \)
4.  **Define Decision Variables:**
    - \( x[i] \) = Number of units of widget \( i \) to produce in the month. Type: GRB.CONTINUOUS (non-negative).
    - \( s \) = Amount (kg) of CatalystX sold in the month. Type: GRB.CONTINUOUS (bounded, non-negative).
    - \( d \) = Amount (kg) of CatalystX disposed of in the month. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'Profit' (from product_resources.csv) for each widget \( i \).
        - Sale price of CatalystX: $300/kg (fixed, not in CSV).
        - Disposal cost of CatalystX: $200/kg (fixed, not in CSV).
    - Constraint coefficients:
        - 'LaborHours', 'MaterialA', 'MaterialB' (from product_resources.csv) per unit of widget \( i \).
        - For Widget3 only: 5 kg CatalystX generated per unit produced.
    - Constraint RHS (limits):
        - 'MonthlyLimit' for 'LaborHours', 'MaterialA', 'MaterialB' (from resource_limits.csv).
        - CatalystX sales cap: 1500 kg/month (fixed, not in CSV).
6.  **Formulate Objective:** Maximize total profit, which is the sum of:
    - Total base profit from all widgets: \( \sum_{i} \text{Profit}[i] \cdot x[i] \)
    - Plus revenue from CatalystX sold: \( +300 \cdot s \)
    - Minus disposal cost for unsold CatalystX: \( -200 \cdot d \)
7.  **Formulate Constraints:**
    - Resource Constraints:
        - Labor: \( \sum_{i} \text{LaborHours}[i] \cdot x[i] \leq \text{MonthlyLimit}[\text{LaborHours}] \)
        - Material A: \( \sum_{i} \text{MaterialA}[i] \cdot x[i] \leq \text{MonthlyLimit}[\text{MaterialA}] \)
        - Material B: \( \sum_{i} \text{MaterialB}[i] \cdot x[i] \leq \text{MonthlyLimit}[\text{MaterialB}] \)
    - CatalystX Byproduct Balance:
        - Total CatalystX generated: \( 5 \cdot x[\text{Widget3}] = s + d \)
    - CatalystX Sales Cap:
        - \( s \leq 1500 \)
    - Non-negativity:
        - \( x[i] \geq 0 \) for all widgets \( i \)
        - \( s \geq 0 \)
        - \( d \geq 0 \)
[Abstract Model Plan END]