[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three grades of raw wine material to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): {I, II, III}
    - Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by grade.
    -   Raw material unit costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by grade.
    -   Brand selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by brand.
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, specifying lower and/or upper bounds on the proportion of certain grades in each brand.
    -   Minimum production requirement: For the Red brand, total production (sum over grades) must be at least 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over all brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    -   Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × (x[g, b])
    -   Objective: Maximize [sum_b (Selling Price_b × sum_g x[g, b])] – [sum_g sum_b (Cost_g × x[g, b])]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the brand’s blend must satisfy the lower and/or upper bounds given in 'Blending Requirements'. For example, for Red: proportion of I < 10%, proportion of II > 50%, etc. This is modeled as:
        -   For each (brand b, grade g) with an upper bound: x[g, b] ≤ (upper%) × total production of brand b
        -   For each (brand b, grade g) with a lower bound: x[g, b] ≥ (lower%) × total production of brand b
        -   Where total production of brand b = sum over grades of x[g, b]
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:
        -   For each grade g: sum over brands b of x[g, b] ≤ Daily Supply (kg) of grade g
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand must be at least 2,000 kg:
        -   sum over grades g of x[g, Red] ≥ 2,000
    -   Constraint 4 (Non-negativity): All x[g, b] ≥ 0
[Abstract Model Plan END]