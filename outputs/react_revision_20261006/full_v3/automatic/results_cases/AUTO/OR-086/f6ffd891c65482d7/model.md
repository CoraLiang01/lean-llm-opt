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
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (parsed into lower/upper bounds on proportions of certain grades in each brand).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    -   Minus total raw material cost: sum over grades of (cost per kg of grade) × (total kg of that grade used, i.e., sum over brands of x[g, b])
    -   Objective: Maximize [sum_b (SellingPrice_b × sum_g x[g, b])] – [sum_g (Cost_g × sum_b x[g, b])]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the brand’s blend must satisfy the lower and/or upper bounds as specified in 'Blending Requirements'. For example, for Red: (x[I, Red]) / (sum_g x[g, Red]) ≤ 0.10, (x[II, Red]) / (sum_g x[g, Red]) ≥ 0.50, etc.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit: sum_b x[g, b] ≤ supply limit for grade g.
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand must be at least 2,000 kg: sum_g x[g, Red] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]