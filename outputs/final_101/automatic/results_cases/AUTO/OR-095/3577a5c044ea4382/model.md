[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production quantities for each of 141 aerospace-grade widgets (Widget1 through Widget141), considering labor, material, and byproduct constraints, to maximize total profit. Special attention is required for Widget3, which generates a valuable byproduct (CatalystX) that can be sold (with a sales cap) or must be disposed of at a cost if unsold.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) resource allocation and byproduct management problem.
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \{\text{Widget1}, \ldots, \text{Widget141}\} \)
4.  **Define Decision Variables:**
    - `x[i]` = Number of units of widget \( i \) to produce in the month. Type: GRB.CONTINUOUS (non-negative real).
    - `s` = Amount (kg) of CatalystX sold in the month. Type: GRB.CONTINUOUS (bounded above by sales cap, non-negative).
    - `d` = Amount (kg) of CatalystX disposed of in the month. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - Widget profit per unit: from product_resources.csv, column 'Profit'.
        - CatalystX sale price: $300/kg (given in query).
        - CatalystX disposal cost: $200/kg (given in query).
    - Constraint coefficients:
        - Labor hours per widget: product_resources.csv, column 'LaborHours'.
        - MaterialA per widget: product_resources.csv, column 'MaterialA'.
        - MaterialB per widget: product_resources.csv, column 'MaterialB'.
        - CatalystX byproduct rate: 5 kg per unit of Widget3 (given in query; zero for other widgets).
    - Constraint RHS (limits):
        - LaborHours: resource_limits.csv, row 'LaborHours', column 'MonthlyLimit' (5000).
        - MaterialA: resource_limits.csv, row 'MaterialA', column 'MonthlyLimit' (24000).
        - MaterialB: resource_limits.csv, row 'MaterialB', column 'MonthlyLimit' (15000).
        - CatalystX sales cap: 1500 kg/month (given in query).
6.  **Formulate Objective:** Maximize total profit, which is:
    - Sum of (profit per unit × units produced) for all widgets
    - Plus revenue from CatalystX sold ($300 × s)
    - Minus disposal cost for unsold CatalystX ($200 × d)
    - That is:  
      Maximize  
      \( \sum_{i=1}^{141} \text{Profit}[i] \cdot x[i] + 300 \cdot s - 200 \cdot d \)
7.  **Formulate Constraints:**
    - **Labor Hours Constraint:**  
      \( \sum_{i=1}^{141} \text{LaborHours}[i] \cdot x[i] \leq 5000 \)
    - **Material A Constraint:**  
      \( \sum_{i=1}^{141} \text{MaterialA}[i] \cdot x[i] \leq 24000 \)
    - **Material B Constraint:**  
      \( \sum_{i=1}^{141} \text{MaterialB}[i] \cdot x[i] \leq 15000 \)
    - **CatalystX Mass Balance:**  
      Total CatalystX generated = 5 × x[Widget3]  
      \( s + d = 5 \cdot x[\text{Widget3}] \)
    - **CatalystX Sales Cap:**  
      \( s \leq 1500 \)
    - **Non-negativity:**  
      \( x[i] \geq 0 \) for all widgets  
      \( s \geq 0 \)  
      \( d \geq 0 \)
[Abstract Model Plan END]