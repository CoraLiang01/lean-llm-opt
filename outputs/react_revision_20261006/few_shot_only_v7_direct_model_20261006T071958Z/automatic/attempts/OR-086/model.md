[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III} (from 30-1.csv)
    - Brands: {Red, Yellow, Blue} (from 30-2.csv)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw grade daily supply limits: from 30-1.csv, column 'Daily Supply (kg)' (per grade).
    -   Raw grade unit costs: from 30-1.csv, column 'Cost (CNY/kg)' (per grade).
    -   Brand selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' (per brand).
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' (per brand, specifying upper/lower bounds on proportions of certain grades).
    -   Minimum production for Red: 2,000 kg (from query).
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    - Minus total raw material cost: sum over grades of (unit cost of grade) × (total kg of that grade used, i.e., sum over brands of x[g, b])
    - Objective: Maximize  
      sum_b [SellingPrice_b × sum_g x[g, b]] − sum_g [Cost_g × sum_b x[g, b]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Raw Material Supply Limits): For each grade g,  
        sum_b x[g, b] ≤ DailySupply_g
    -   Constraint 2 (Blending Requirements): For each brand b and each specified grade g in the blending requirements for b:
        - If requirement is "g less than α%":  
          x[g, b] ≤ (α/100) × sum_{g'} x[g', b]
        - If requirement is "g more than β%":  
          x[g, b] ≥ (β/100) × sum_{g'} x[g', b]
        (Apply these for each brand as specified in 30-2.csv.)
    -   Constraint 3 (Minimum Production for Red):  
        sum_g x[g, Red] ≥ 2,000
    -   Constraint 4 (Non-negativity):  
        x[g, b] ≥ 0 for all grades g and brands b
[Abstract Model Plan END]