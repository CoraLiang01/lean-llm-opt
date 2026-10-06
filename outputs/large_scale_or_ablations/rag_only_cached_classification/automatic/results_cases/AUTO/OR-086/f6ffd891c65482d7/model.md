[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three wine brands by blending three grades of raw wine, subject to blending requirements, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): {I, II, III}
    - Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv:
        -   `Supply[g]`: Daily supply limit for grade `g` ('Daily Supply (kg)')
        -   `Cost[g]`: Unit cost for grade `g` ('Cost (CNY/kg)')
    -   From 30-2.csv:
        -   `Price[b]`: Selling price per kg for brand `b` ('Selling Price (CNY/kg)')
        -   `BlendReq[b]`: Blending requirements for each brand `b` ('Blending Requirements'), parsed into lower and upper bounds on the proportion of certain grades in each brand.
    -   The minimum production requirement for the Red brand: 2,000 kg (from query).
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        Total sales revenue (sum over brands of total kg produced × selling price) 
        minus 
        Total raw material cost (sum over all grades and brands of amount used × unit cost).
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the total blend for the brand must satisfy the lower and/or upper bounds as specified in 'Blending Requirements'. For example, for Red: proportion of I < 10%, proportion of II > 50%, etc.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit: sum over brands of x[g, b] ≤ Supply[g].
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand must be at least 2,000 kg: sum over grades of x[g, 'Red'] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All decision variables x[g, b] ≥ 0.
[Abstract Model Plan END]