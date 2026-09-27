[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material: {I, II, III}
    - Wine brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: from 30-1.csv, column 'Daily Supply (kg)' for each grade.
    -   Raw material costs: from 30-1.csv, column 'Cost (CNY/kg)' for each grade.
    -   Selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' for each brand.
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (e.g., upper/lower bounds on proportions of certain grades in each brand).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    - Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × x[g, b]
    - Objective: Maximize [sum_b (SellingPrice_b × sum_g x[g, b])] − [sum_g sum_b (Cost_g × x[g, b])]
7.  **Formulate Constraints:**
    -   Constraint 1 (Raw Material Supply Limits): For each grade g, sum over all brands b of x[g, b] ≤ supply limit for grade g.
    -   Constraint 2 (Blending Requirements): For each brand b and each specified grade g in the blending requirements:
        - For upper-bound: x[g, b] / (sum over all grades g' of x[g', b]) ≤ specified maximum proportion.
        - For lower-bound: x[g, b] / (sum over all grades g' of x[g', b]) ≥ specified minimum proportion.
        - (These are ratio constraints, implemented as linear inequalities by cross-multiplying.)
    -   Constraint 3 (Minimum Production for Red): sum over all grades g of x[g, Red] ≥ 2,000 kg.
    -   Constraint 4 (Non-negativity): x[g, b] ≥ 0 for all grades g and brands b.
[Abstract Model Plan END]