[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): {I, II, III}
    - Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by Grade.
    -   Raw material unit costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by Grade.
    -   Selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by Brand.
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, parsed for each brand to extract lower and upper bounds on the proportion of certain grades in each brand.
    -   Minimum production requirement: Red brand must have at least 2,000 kg produced per day.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over all brands of (selling price of brand) × (total kg produced of that brand)
    -   Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × (kg of grade used in brand)
    -   In formula: Maximize sum_b [SellingPrice[b] × sum_g x[g, b]] − sum_g sum_b [Cost[g] × x[g, b]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade in its blending requirements, the proportion of that grade in the total blend for the brand must satisfy the given lower and/or upper bound. For example, for Red: (x[I, Red] / total_Red) ≤ 10%, (x[II, Red] / total_Red) ≥ 50%, where total_Red = sum_g x[g, Red]. Similar constraints for Yellow and Blue, as parsed from their blending requirements.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit. For each grade g: sum_b x[g, b] ≤ DailySupply[g].
    -   Constraint 3 (Minimum Production for Red): The total production of Red brand must be at least 2,000 kg: sum_g x[g, Red] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0.
[Abstract Model Plan END]