[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three grades of raw wine to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): {I, II, III}
    - Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by Grade.
    -   Raw material costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by Grade.
    -   Selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by Brand.
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, parsed into lower and upper bounds for the proportion of certain grades in each brand.
    -   Minimum production requirement: Red brand must have at least 2,000 kg produced per day.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over all brands of (total kg produced of brand b) × (selling price of brand b)
    -   Minus total raw material cost: sum over all grades and brands of (kg of grade g used in brand b) × (cost per kg of grade g)
    -   In formula: Maximize sum_b [ (sum_g x[g, b]) × price[b] ] - sum_g sum_b [ x[g, b] × cost[g] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the brand’s blend must satisfy the lower and/or upper bounds as specified in 'Blending Requirements'. For example, for Red: proportion of I < 10%, proportion of II > 50%, etc. This is implemented as:
        -   For each (brand b, grade g) with a requirement:  
            -   Lower bound: x[g, b] ≥ lower_bound × total_production_of_b  
            -   Upper bound: x[g, b] ≤ upper_bound × total_production_of_b  
            where total_production_of_b = sum_g x[g, b]
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:
        -   For each grade g: sum_b x[g, b] ≤ supply_limit[g]
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand must be at least 2,000 kg:
        -   sum_g x[g, 'Red'] ≥ 2,000
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0
[Abstract Model Plan END]