[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem (continuous variables, linear objective and constraints).
3.  **Define Index Sets:** The primary index is Foods (set of all foods listed in the CSV, 120 items).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food `i` to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients: 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)' columns (nutrient content per serving for each food).
    -   Constraint RHS (limits): 2000 (min Calories), 50 (min Protein), 60 (min Vitamin C), 70 (max Fat).
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings chosen):  
    Minimize: sum over i of [Cost[i] * x[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum): The total calories from all selected foods must be at least 2000.  
        sum over i of [Calories[i] * x[i]] ≥ 2000
    -   Constraint 2 (Protein Minimum): The total protein from all selected foods must be at least 50 grams.  
        sum over i of [Protein(g)[i] * x[i]] ≥ 50
    -   Constraint 3 (Vitamin C Minimum): The total vitamin C from all selected foods must be at least 60 mg.  
        sum over i of [VitaminC(mg)[i] * x[i]] ≥ 60
    -   Constraint 4 (Fat Maximum): The total fat from all selected foods must not exceed 70 grams.  
        sum over i of [Fat(g)[i] * x[i]] ≤ 70
    -   Constraint 5 (Non-negativity):  
        x[i] ≥ 0 for all foods i
[Abstract Model Plan END]