[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (Grade I, II, III)
    - Wine brands (Red, Yellow, Blue)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits and costs from 30-1.csv:
        - 'Daily Supply (kg)' for each Grade (I, II, III)
        - 'Cost (CNY/kg)' for each Grade
    -   Selling prices and blending requirements from 30-2.csv:
        - 'Selling Price (CNY/kg)' for each Brand (Red, Yellow, Blue)
        - 'Blending Requirements' for each Brand (e.g., "I less than 10%", "II more than 50%")
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (total kg produced of each brand) × (selling price of that brand)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade used in brand) × (cost per kg of that grade)
    - In formula:  
      Maximize  
      sum_b [ (sum_g x[g, b]) × SellingPrice[b] ]  
      − sum_g sum_b [ x[g, b] × Cost[g] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce upper and/or lower bounds on the proportion of that grade in the blend:
        - For example, for Red:  
            - Proportion of Grade I in Red < 10%:  
              x[I, Red] / (sum_g x[g, Red]) ≤ 0.10
            - Proportion of Grade II in Red > 50%:  
              x[II, Red] / (sum_g x[g, Red]) ≥ 0.50
        - Similarly, apply all blending requirements for Yellow and Blue as specified in 30-2.csv.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit:
        - For each grade g:  
          sum_b x[g, b] ≤ DailySupply[g]
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg:
        - sum_g x[g, Red] ≥ 2000
    -   Constraint 4 (Non-negativity): All decision variables must be non-negative:
        - x[g, b] ≥ 0 for all grades g and brands b
[Abstract Model Plan END]