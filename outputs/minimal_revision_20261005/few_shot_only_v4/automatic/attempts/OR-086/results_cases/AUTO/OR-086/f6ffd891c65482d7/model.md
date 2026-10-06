[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III}
    - Brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv (all rows required):
        - `Supply[g]`: Daily supply limit for grade `g` ("Daily Supply (kg)")
        - `Cost[g]`: Unit cost for grade `g` ("Cost (CNY/kg)")
    -   From 30-2.csv (all rows required):
        - `Price[b]`: Selling price for brand `b` ("Selling Price (CNY/kg)")
        - `BlendReq[b]`: Blending requirements for brand `b` ("Blending Requirements")
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        sum over all brands b [ sum over all grades g (Price[b] * x[g, b]) ] 
        minus 
        sum over all grades g [ sum over all brands b (Cost[g] * x[g, b]) ]
    That is, maximize total sales revenue from all brands minus total raw material cost.
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements for Each Brand): For each brand, the proportion of selected grades in the blend must satisfy specified upper and lower bounds. For example:
        - For Red: 
            - Proportion of Grade I in Red < 10%: x[I, Red] / total_Red < 0.10
            - Proportion of Grade II in Red > 50%: x[II, Red] / total_Red > 0.50
        - For Yellow:
            - Proportion of Grade III in Yellow < 70%: x[III, Yellow] / total_Yellow < 0.70
            - Proportion of Grade I in Yellow > 20%: x[I, Yellow] / total_Yellow > 0.20
        - For Blue:
            - Proportion of Grade I in Blue < 50%: x[I, Blue] / total_Blue < 0.50
            - Proportion of Grade II in Blue > 10%: x[II, Blue] / total_Blue > 0.10
        (Where total_{Brand} = sum over g of x[g, Brand]; if total_{Brand} = 0, proportions are undefined, so model should ensure total_{Brand} ≥ 0.)
    -   Constraint 2 (Raw Material Supply Limits): For each grade, the total amount used across all brands cannot exceed its daily supply:
        - For each grade g: sum over brands b of x[g, b] ≤ Supply[g]
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg:
        - sum over grades g of x[g, Red] ≥ 2,000
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0
[Abstract Model Plan END]