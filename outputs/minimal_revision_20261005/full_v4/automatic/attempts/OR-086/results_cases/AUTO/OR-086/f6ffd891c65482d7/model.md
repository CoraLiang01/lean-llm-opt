[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (Grade I, II, III)
    - Wine brands (Red, Yellow, Blue)
4.  **Define Decision Variables:**
    -   `x[b, g]` = Amount (kg) of raw grade `g` used in brand `b` per day. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv:
        - `Supply[g]`: Daily supply limit for grade `g` ('Daily Supply (kg)')
        - `Cost[g]`: Unit cost for grade `g` ('Cost (CNY/kg)')
    -   From 30-2.csv:
        - `Price[b]`: Selling price per kg for brand `b` ('Selling Price (CNY/kg)')
        - `BlendingReq[b]`: Blending requirements for brand `b` ('Blending Requirements'), specifying upper/lower bounds on the proportion of certain grades in each brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (total kg produced of brand `b`) × (selling price of brand `b`)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade `g` used in brand `b`) × (cost of grade `g`)
    - In formula: Maximize  
      sum_b [ Price[b] × sum_g x[b, g] ] − sum_b sum_g [ Cost[g] × x[b, g] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the blend must satisfy the given upper or lower bound. For example, for Red:  
        - Proportion of Grade I in Red < 10%: x[Red, I] / (sum_g x[Red, g]) ≤ 0.10  
        - Proportion of Grade II in Red > 50%: x[Red, II] / (sum_g x[Red, g]) ≥ 0.50  
        - Similarly, apply the blending requirements for Yellow and Blue as specified in 'Blending Requirements'.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:  
        - sum_b x[b, g] ≤ Supply[g]  for all grades g
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg:  
        - sum_g x[Red, g] ≥ 2000
    -   Constraint 4 (Non-negativity): All decision variables must be non-negative:  
        - x[b, g] ≥ 0 for all brands b and grades g
[Abstract Model Plan END]