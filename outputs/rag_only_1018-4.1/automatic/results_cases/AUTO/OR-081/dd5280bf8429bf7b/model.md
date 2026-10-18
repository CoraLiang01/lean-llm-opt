[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of available foods, denoted as `i ∈ Foods` (where Foods is the set of all food items listed in the CSV).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0 (x[i] ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint right-hand sides (RHS):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings):  
    Minimize: sum over i of [Cost[i] * x[i]]
7.  **Formulate Constraints:**
    -   Calorie constraint: The total calories from all selected servings must be at least 2000 kcal:  
        sum over i of [Calories[i] * x[i]] ≥ 2000
    -   Protein constraint: The total protein from all selected servings must be at least 50 g:  
        sum over i of [Protein(g)[i] * x[i]] ≥ 50
    -   Vitamin C constraint: The total vitamin C from all selected servings must be at least 60 mg:  
        sum over i of [VitaminC(mg)[i] * x[i]] ≥ 60
    -   Fat constraint: The total fat from all selected servings must not exceed 70 g:  
        sum over i of [Fat(g)[i] * x[i]] ≤ 70
    -   Non-negativity: For all foods i, x[i] ≥ 0
[Abstract Model Plan END]