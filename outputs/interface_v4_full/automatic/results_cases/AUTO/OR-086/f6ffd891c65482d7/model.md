[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
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
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (parsed into lower/upper bounds on proportions of certain grades in each brand).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, i.e.,  
    - Total sales revenue: sum over brands of (selling price of brand) × (total kg produced of that brand, i.e., sum over grades of x[g, b])
    - Minus total raw material cost: sum over grades and brands of (cost per kg of grade) × (x[g, b])
    - Objective: Maximize  
      sum_b [SellingPrice_b × sum_g x[g, b]] − sum_g sum_b [Cost_g × x[g, b]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the brand's blend must satisfy the lower and/or upper bounds given in 'Blending Requirements'. For example, for Red:  
        - Proportion of I in Red < 10%: x[I, Red] / (sum_g x[g, Red]) ≤ 0.10  
        - Proportion of II in Red > 50%: x[II, Red] / (sum_g x[g, Red]) ≥ 0.50  
        (Similar constraints for Yellow and Blue, as parsed from their requirements.)
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:  
        sum_b x[g, b] ≤ DailySupply_g, for all g ∈ {I, II, III}
    -   Constraint 3 (Minimum Production for Red):  
        sum_g x[g, Red] ≥ 2,000 kg
    -   Constraint 4 (Non-negativity):  
        x[g, b] ≥ 0 for all grades and brands
[Abstract Model Plan END]