[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (G): {I, II, III}
    - Wine brands (B): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: from 30-1.csv, column 'Daily Supply (kg)' for each grade.
    -   Raw material costs: from 30-1.csv, column 'Cost (CNY/kg)' for each grade.
    -   Selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' for each brand.
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (specifying upper/lower bounds on proportions of certain grades).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        sum over brands b [ (selling price of b) × (total kg produced of b) ] 
        minus 
        sum over grades g [ (cost of g) × (total kg of g used across all brands) ].
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand b and each specified grade g, enforce lower and/or upper bounds on the proportion of grade g in the total blend for brand b, as specified in 'Blending Requirements' (e.g., x[g, b] / sum_g' x[g', b] ≤ upper_bound, x[g, b] / sum_g' x[g', b] ≥ lower_bound).
    -   Constraint 2 (Raw Material Supply): For each grade g, the total amount used across all brands does not exceed its daily supply limit (sum_b x[g, b] ≤ supply limit for g).
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand is at least 2,000 kg (sum_g x[g, 'Red'] ≥ 2,000).
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]