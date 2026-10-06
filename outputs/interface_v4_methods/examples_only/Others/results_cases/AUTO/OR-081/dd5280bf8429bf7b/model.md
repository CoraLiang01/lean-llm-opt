[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem (continuous variables, linear objective and constraints).
3.  **Define Index Sets:** The primary index is Foods (i), where each food is a row in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of each food).
    -   Constraint coefficients:
        -   'Calories' (calories per serving)
        -   'Protein(g)' (protein per serving)
        -   'Fat(g)' (fat per serving)
        -   'VitaminC(mg)' (vitamin C per serving)
    -   Constraint RHS (limits): Provided in the query:
        -   Calories: ≥ 2000
        -   Protein: ≥ 50 g
        -   Vitamin C: ≥ 60 mg
        -   Fat: ≤ 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan: sum over all foods of (Cost per serving) × (number of servings of that food), i.e., minimize sum_i Cost[i] * x[i].
7.  **Formulate Constraints:**
    -   Constraint 1 (Calories): The total calories from all selected foods must be at least 2000: sum_i Calories[i] * x[i] ≥ 2000.
    -   Constraint 2 (Protein): The total protein from all selected foods must be at least 50 g: sum_i Protein[i] * x[i] ≥ 50.
    -   Constraint 3 (Vitamin C): The total vitamin C from all selected foods must be at least 60 mg: sum_i VitaminC[i] * x[i] ≥ 60.
    -   Constraint 4 (Fat): The total fat from all selected foods must not exceed 70 g: sum_i Fat[i] * x[i] ≤ 70.
    -   Constraint 5 (Non-negativity): x[i] ≥ 0 for all foods i.
[Abstract Model Plan END]