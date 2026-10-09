[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades (G): {I, II, III} (from 30-1.csv)
    - Brands (B): {Red, Yellow, Blue} (from 30-2.csv)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Raw grade daily supply limits: 'Daily Supply (kg)' from 30-1.csv.
    -   Raw grade unit costs: 'Cost (CNY/kg)' from 30-1.csv.
    -   Brand selling prices: 'Selling Price (CNY/kg)' from 30-2.csv.
    -   Blending requirements: 'Blending Requirements' from 30-2.csv (specifies upper/lower bounds on proportions of certain grades in each brand).
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        - Total sales revenue: sum over all brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
        - Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × x[g, b]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce the upper and/or lower bound on the proportion of that grade in the total blend for that brand. For example, for brand b and grade g with a requirement "g less than α%": x[g, b] ≤ (α/100) × total production of brand b; for "g more than β%": x[g, b] ≥ (β/100) × total production of brand b.
    -   Constraint 2 (Raw Material Supply): For each grade g, the total amount used across all brands cannot exceed its daily supply limit: sum over brands of x[g, b] ≤ supply limit for grade g.
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand (sum over grades of x[g, Red]) must be at least 2,000 kg: sum over grades of x[g, Red] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]