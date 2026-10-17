[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements for each brand, raw material daily supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III}
    - Brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Raw grade daily supply limits: from 30-1.csv, column 'Daily Supply (kg)' (per grade).
    -   Raw grade unit costs: from 30-1.csv, column 'Cost (CNY/kg)' (per grade).
    -   Brand selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' (per brand).
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' (per brand, specifying upper/lower bounds on proportions of certain grades).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        sum over brands b [ (selling price of b) × (total kg produced of b) ] 
        minus 
        sum over grades g and brands b [ (cost of g) × x[g, b] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce the stated upper or lower bound on the proportion of that grade in the total blend for that brand. For example, for Red: proportion of I < 10%, proportion of II > 50%; for Yellow: proportion of III < 70%, proportion of I > 20%; for Blue: proportion of I < 50%, proportion of II > 10%. Express as: x[g, b] / sum_g' x[g', b] ≤ or ≥ bound, for all required (g, b) pairs, with sum_g' x[g', b] > 0 if the brand is produced.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit: sum_b x[g, b] ≤ supply limit for g.
    -   Constraint 3 (Minimum Production for Red): The total daily production of Red must be at least 2,000 kg: sum_g x[g, Red] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]