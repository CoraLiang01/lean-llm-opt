[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three grades of raw wine material to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
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
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, specifying upper and lower bounds on the proportion of certain grades in each brand.
    -   Minimum production requirement: Red brand must have at least 2,000 kg produced per day.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (total kg produced of brand) × (brand selling price)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade used in brand) × (grade unit cost)
    - In formula: Maximize  
      sum_b [ (sum_g x[g, b]) × SellingPrice[b] ]  
      minus  
      sum_g sum_b [ x[g, b] × Cost[g] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the total blend for the brand must satisfy the given upper or lower bound. For example, for Red:  
        - Proportion of I in Red ≤ 10%  
        - Proportion of II in Red ≥ 50%  
        (Proportion is x[g, b] / sum_g x[g, b]; implement as linear constraints using auxiliary variables or by multiplying both sides by sum_g x[g, b].)
    -   Constraint 2 (Raw Material Supply Limits): For each grade, the total amount used across all brands cannot exceed its daily supply limit.  
        - For each grade g: sum_b x[g, b] ≤ DailySupply[g]
    -   Constraint 3 (Minimum Production for Red):  
        - sum_g x[g, 'Red'] ≥ 2,000
    -   Constraint 4 (Non-negativity):  
        - x[g, b] ≥ 0 for all grades g and brands b
[Abstract Model Plan END]