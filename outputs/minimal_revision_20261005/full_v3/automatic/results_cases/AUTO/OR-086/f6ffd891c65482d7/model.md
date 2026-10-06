[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material: {I, II, III}
    - Wine brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: from 30-1.csv, column 'Daily Supply (kg)' for each grade.
    -   Raw material costs: from 30-1.csv, column 'Cost (CNY/kg)' for each grade.
    -   Selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' for each brand.
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (e.g., "I less than 10%", "II more than 50%").
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        - Total sales revenue: sum over all brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
        - Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × x[g, b]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce the upper and/or lower bound on the proportion of that grade in the total blend for that brand. For example, for Red: 
        - Proportion of Grade I in Red < 10%: x[I, Red] / (sum over g of x[g, Red]) ≤ 0.10
        - Proportion of Grade II in Red > 50%: x[II, Red] / (sum over g of x[g, Red]) ≥ 0.50
        - Similarly for Yellow and Blue, as specified in 'Blending Requirements'.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:
        - sum over brands of x[g, b] ≤ supply limit for grade g
    -   Constraint 3 (Minimum Production for Red): The total production of Red must be at least 2,000 kg:
        - sum over grades of x[g, Red] ≥ 2,000
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0
[Abstract Model Plan END]