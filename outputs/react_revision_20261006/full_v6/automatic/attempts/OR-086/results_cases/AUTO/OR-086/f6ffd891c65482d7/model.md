[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (Grade I, II, III)
    - Wine brands (Red, Yellow, Blue)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv:
        - `Supply[g]`: Daily supply limit for grade `g` ('Daily Supply (kg)')
        - `Cost[g]`: Unit cost for grade `g` ('Cost (CNY/kg)')
    -   From 30-2.csv:
        - `Price[b]`: Selling price per kg for brand `b` ('Selling Price (CNY/kg)')
        - `BlendReq[b]`: Blending requirements for brand `b` ('Blending Requirements'), specifying upper/lower bounds on the proportion of certain grades in the blend.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over brands of (total kg produced of brand `b`) × (selling price of brand `b`)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade `g` used in brand `b`) × (cost of grade `g`)
    - In formula: Maximize  
      sum_b [ Price[b] × sum_g x[g, b] ] − sum_g sum_b [ Cost[g] × x[g, b] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce the upper and/or lower bound on the proportion of that grade in the total blend for that brand. For example, for Red:  
        - Proportion of Grade I in Red < 10%: x[I, Red] ≤ 0.10 × total_Red  
        - Proportion of Grade II in Red > 50%: x[II, Red] ≥ 0.50 × total_Red  
        (where total_Red = sum_g x[g, Red]).  
        Similarly, apply all blending requirements for Yellow and Blue as specified in the 'Blending Requirements' column.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:  
        sum_b x[g, b] ≤ Supply[g]  for all grades g
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg:  
        sum_g x[g, Red] ≥ 2,000
    -   Constraint 4 (Non-negativity):  
        x[g, b] ≥ 0 for all grades g and brands b
[Abstract Model Plan END]