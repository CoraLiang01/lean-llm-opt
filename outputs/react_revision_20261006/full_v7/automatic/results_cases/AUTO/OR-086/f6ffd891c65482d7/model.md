[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): {I, II, III}
    - Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by grade.
    -   Raw material costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by grade.
    -   Selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by brand.
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, specifying upper/lower bounds on the proportion of certain grades in each brand.
    -   Minimum production for Red: 2,000 kg (from query).
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    -   Minus total raw material cost: sum over grades and brands of (cost per kg of grade) × x[g, b]
    -   Objective: Maximize sum_b [SellingPrice_b × sum_g x[g, b]] − sum_g sum_b [Cost_g × x[g, b]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Raw Material Supply Limits): For each grade g, sum over brands b of x[g, b] ≤ Daily Supply (kg) for grade g.
    -   Constraint 2 (Blending Requirements): For each brand b and each specified grade g in the blending requirements:
        -   If requirement is "g less than α%": x[g, b] ≤ (α/100) × total production of brand b (i.e., sum_g' x[g', b])
        -   If requirement is "g more than β%": x[g, b] ≥ (β/100) × total production of brand b
        -   (Apply all such upper/lower bounds as specified for each brand in 30-2.csv)
    -   Constraint 3 (Minimum Production for Red): sum_g x[g, 'Red'] ≥ 2,000
    -   Constraint 4 (Non-negativity): x[g, b] ≥ 0 for all grades g and brands b
[Abstract Model Plan END]