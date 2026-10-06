[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), subject to blending requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III}
    - Brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real variables).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv (all rows required):
        - `Supply[g]`: Daily supply limit for raw grade `g` ("Daily Supply (kg)")
        - `Cost[g]`: Unit cost for raw grade `g` ("Cost (CNY/kg)")
    -   From 30-2.csv (all rows required):
        - `Price[b]`: Selling price per kg for brand `b` ("Selling Price (CNY/kg)")
        - `BlendReq[b]`: Blending requirements for brand `b` ("Blending Requirements"), specifying upper/lower bounds on proportions of certain grades in each brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (total kg produced of brand `b`) × (selling price of brand `b`)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade `g` used in brand `b`) × (cost of grade `g`)
    - In formula: Maximize  
      sum_b [Price[b] × sum_g x[g, b]] − sum_g sum_b [Cost[g] × x[g, b]]
7.  **Formulate Constraints:**
    -   **Constraint 1 (Blending Requirements):** For each brand, enforce the specified upper/lower bounds on the proportion of certain grades in the blend:
        - For Red: Proportion of I < 10%; Proportion of II > 50%
        - For Yellow: Proportion of III < 70%; Proportion of I > 20%
        - For Blue: Proportion of I < 50%; Proportion of II > 10%
        - For each brand `b`, let total production be S_b = sum_g x[g, b]. For each requirement, e.g., "I less than 10%": x[I, b] ≤ 0.10 × S_b; "II more than 50%": x[II, b] ≥ 0.50 × S_b, etc.
    -   **Constraint 2 (Raw Material Supply Limits):** For each grade, the total amount used across all brands cannot exceed its daily supply:
        - For each grade `g`: sum_b x[g, b] ≤ Supply[g]
    -   **Constraint 3 (Minimum Production for Red):** The total daily production of the Red brand must be at least 2,000 kg:
        - sum_g x[g, Red] ≥ 2,000
    -   **Constraint 4 (Non-negativity):** All decision variables must be non-negative:
        - x[g, b] ≥ 0 for all grades `g` and brands `b`
[Abstract Model Plan END]