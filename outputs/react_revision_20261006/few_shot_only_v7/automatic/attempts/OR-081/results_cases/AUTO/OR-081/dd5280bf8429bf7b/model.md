[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (each row in the cost.csv table).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings (can be fractional) of food `f` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings of that food):  
    Minimize: sum_f [ Cost[f] * x[f] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum): The total calories from all selected foods must be at least 2000 kcal.  
        sum_f [ Calories[f] * x[f] ] ≥ 2000
    -   Constraint 2 (Protein Minimum): The total protein from all selected foods must be at least 50 g.  
        sum_f [ Protein(g)[f] * x[f] ] ≥ 50
    -   Constraint 3 (Vitamin C Minimum): The total vitamin C from all selected foods must be at least 60 mg.  
        sum_f [ VitaminC(mg)[f] * x[f] ] ≥ 60
    -   Constraint 4 (Fat Maximum): The total fat from all selected foods must not exceed 70 g.  
        sum_f [ Fat(g)[f] * x[f] ] ≤ 70
    -   Constraint 5 (Non-negativity):  
        For all foods f: x[f] ≥ 0
[Abstract Model Plan END]