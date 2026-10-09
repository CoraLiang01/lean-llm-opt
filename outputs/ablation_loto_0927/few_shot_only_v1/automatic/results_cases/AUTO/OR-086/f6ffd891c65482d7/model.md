[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), subject to blending requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III} (from 30-1.csv)
    - Brands: {Red, Yellow, Blue} (from 30-2.csv)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Raw grade daily supply limits: from 30-1.csv, column 'Daily Supply (kg)' (per grade).
    -   Raw grade unit costs: from 30-1.csv, column 'Cost (CNY/kg)' (per grade).
    -   Brand selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' (per brand).
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' (per brand, parsed for each grade).
    -   Minimum production for Red: 2,000 kg (from query).
6.  **Formulate Objective:** Maximize total net profit, i.e., total sales revenue from all brands minus total raw material costs:
    - Total sales revenue = sum over brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    - Total raw material cost = sum over grades of (unit cost of grade) × (total kg of that grade used, i.e., sum over brands of x[g, b])
    - Objective: Maximize [sum_b (SellingPrice_b × sum_g x[g, b])] – [sum_g (Cost_g × sum_b x[g, b])]
7.  **Formulate Constraints:**
    -   Constraint 1 (Raw Material Supply Limits): For each grade g, sum over brands of x[g, b] ≤ daily supply limit of grade g.
    -   Constraint 2 (Blending Requirements): For each brand b and each specified grade g:
        - If a lower bound is specified (e.g., "II more than 50%"), require: x[g, b] ≥ (lower bound %) × (total production of brand b)
        - If an upper bound is specified (e.g., "I less than 10%"), require: x[g, b] ≤ (upper bound %) × (total production of brand b)
        - (Parse and apply the specific requirements for each brand as given in 30-2.csv)
    -   Constraint 3 (Minimum Production for Red): sum over grades of x[g, 'Red'] ≥ 2,000 kg.
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]